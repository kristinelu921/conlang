"""Evaluate the pinned SmolLM2 checkpoint with lm-eval prompts and TPU generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoConfig, AutoTokenizer, FlaxLlamaForCausalLM
from transformers.models.llama import modeling_flax_llama


class ConfiguredRotaryEmbedding(modeling_flax_llama.FlaxLlamaRotaryEmbedding):
    """Honor the checkpoint's RoPE base, which the stock Flax implementation ignores."""

    def setup(self) -> None:
        # SmolLM2: When Smol Goes Big (https://arxiv.org/abs/2502.02737); preserve its checkpoint's RoPE base.
        head_dimension = self.config.hidden_size // self.config.num_attention_heads
        inverse_frequencies = 1.0 / (
            np.float32(self.config.rope_theta)
            ** (np.arange(0, head_dimension, 2, dtype=np.float32) / head_dimension)
        )
        frequencies = np.outer(
            np.arange(self.config.max_position_embeddings, dtype=np.float32),
            inverse_frequencies,
        )
        angles = np.concatenate((frequencies, frequencies), axis=-1)
        self.sincos = jnp.asarray(
            np.concatenate((np.sin(angles)[:, None, :], np.cos(angles)[:, None, :]), axis=-1)
        )


def trim_at_stop(text: str, stop_strings: list[str]) -> str:
    """Return the response before the first configured stop string."""
    endpoints = [text.index(stop) for stop in stop_strings if stop and stop in text]
    return text[: min(endpoints)] if endpoints else text


@register_model("hf-flax-tpu")
class FlaxTPULanguageModel(LM):
    """Keep harness sampling and scoring unchanged while generating on TPU devices."""

    def __init__(
        self,
        pretrained: str,
        revision: str,
        batch_size: int | str = 8,
        device: str = "tpu",
        dtype: str = "float32",
        max_batch_size: int | None = None,
        progress_path: str = "generations-progress.jsonl",
    ) -> None:
        super().__init__()
        if device != "tpu" or dtype != "float32":
            raise ValueError("This benchmark requires TPU inference with float32 computation.")
        self.devices = jax.local_devices(backend="tpu")
        if not self.devices or any(item.platform != "tpu" for item in self.devices):
            raise RuntimeError("Remote TPU devices are required; CPU fallback is disabled.")
        jax.config.update("jax_default_matmul_precision", "highest")
        self._device = "tpu"
        self.pretrained = pretrained
        self.revision = revision
        self.batch_size = int(batch_size)
        self.progress_path = Path(progress_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, revision=revision)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = AutoConfig.from_pretrained(pretrained, revision=revision)
        if self.config.rope_scaling is not None or not self.config.tie_word_embeddings:
            raise ValueError("This adapter supports the pinned, unscaled, tied-embedding SmolLM2 checkpoint.")
        modeling_flax_llama.FlaxLlamaRotaryEmbedding = ConfiguredRotaryEmbedding
        self.model = FlaxLlamaForCausalLM.from_pretrained(
            pretrained, revision=revision, from_pt=True, dtype=jnp.float32
        )
        parameters = self.model.params
        assert isinstance(parameters, dict)
        if set(self.model._missing_keys) - {("lm_head", "kernel")}:
            raise ValueError(f"Unexpected missing checkpoint weights: {self.model._missing_keys}")
        # The stock Flax Llama head is untied; reuse the checkpoint's input embedding exactly.
        parameters["lm_head"] = {"kernel": parameters["model"]["embed_tokens"]["embedding"].T}
        self.parameters = jax.tree.map(lambda value: np.asarray(value, dtype=np.float32), parameters)
        self.replicated_parameters = jax.device_put_replicated(self.parameters, self.devices)
        print(json.dumps({"event": "model_ready", "devices": [str(item) for item in self.devices],
                          "model_revision": revision, "rope_theta": self.config.rope_theta,
                          "batch_per_device": self.batch_size}), flush=True)

    @property
    def tokenizer_name(self) -> str:
        return self.pretrained.replace("/", "__")

    def get_model_info(self) -> dict[str, Any]:
        """Record the checkpoint identity and effective accelerator configuration."""
        return {
            "model_revision": self.revision,
            "model_dtype": "float32",
            "model_num_parameters": 134515008,
            "tpu_devices": [str(item) for item in self.devices],
            "global_batch_size": self.batch_size * len(self.devices),
            "matmul_precision": "highest",
            "rope_theta": self.config.rope_theta,
            "tied_embeddings": True,
        }

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("This benchmark adapter supports generation tasks only.")

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError("This benchmark adapter supports generation tasks only.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate fixed-shape TPU batches and return responses in harness request order."""
        if not requests:
            return []
        settings = requests[0].args[1]
        if any(request.args[1] != settings for request in requests):
            raise ValueError("All requests must use the same generation settings.")
        if settings.get("do_sample", False):
            raise ValueError("This benchmark uses greedy decoding only.")
        generated_token_limit = int(settings.get("max_gen_toks", 256))
        stop_strings = settings["until"]
        tokenized_prompts = [
            self.tokenizer.encode(request.args[0], add_special_tokens=False) for request in requests
        ]
        prompt_width = ((max(map(len, tokenized_prompts)) + 127) // 128) * 128
        if prompt_width + generated_token_limit > self.config.max_position_embeddings:
            raise ValueError("A benchmark prompt exceeds the model context; truncation is forbidden.")
        global_batch_size = self.batch_size * len(self.devices)

        def generate_on_device(
            parameters: dict[str, Any], input_ids: jax.Array, attention_mask: jax.Array
        ) -> jax.Array:
            """Compile greedy cached generation with fixed input and output shapes."""
            return self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                params=parameters,
                max_new_tokens=generated_token_limit,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).sequences

        parallel_generate = jax.pmap(generate_on_device, devices=self.devices)
        responses: list[str] = []
        start_time = time.monotonic()
        print(json.dumps({"event": "generation_started", "total": len(requests),
                          "prompt_width": prompt_width, "max_prompt_tokens": max(map(len, tokenized_prompts)),
                          "global_batch_size": global_batch_size}), flush=True)
        with self.progress_path.open("x") as progress:
            for start in range(0, len(requests), global_batch_size):
                prompts = tokenized_prompts[start : start + global_batch_size]
                input_ids = np.full(
                    (global_batch_size, prompt_width), self.tokenizer.pad_token_id, dtype=np.int32
                )
                attention_mask = np.zeros_like(input_ids)
                for index in range(global_batch_size):
                    tokens = prompts[min(index, len(prompts) - 1)]
                    input_ids[index, -len(tokens) :] = tokens
                    attention_mask[index, -len(tokens) :] = 1
                shape = (len(self.devices), self.batch_size, prompt_width)
                batch_start = time.monotonic()
                generated = np.asarray(parallel_generate(
                    self.replicated_parameters, input_ids.reshape(shape), attention_mask.reshape(shape)
                )).reshape(global_batch_size, -1)
                batch_seconds = time.monotonic() - batch_start
                decoded = self.tokenizer.batch_decode(
                    generated[: len(prompts), prompt_width:], skip_special_tokens=False
                )
                for offset, text in enumerate(decoded):
                    # Detect special-token stops before removing special tokens from the response.
                    response = trim_at_stop(text, stop_strings + [self.tokenizer.eos_token])
                    for special_token in self.tokenizer.all_special_tokens:
                        response = response.replace(special_token, "")
                    responses.append(response)
                    progress.write(json.dumps({"request_index": start + offset, "response": response}) + "\n")
                progress.flush()
                print(json.dumps({"event": "batch_complete", "completed": len(responses),
                                  "total": len(requests), "batch_seconds": batch_seconds,
                                  "elapsed_seconds": time.monotonic() - start_time}), flush=True)
        return responses


if __name__ == "__main__":
    cli_evaluate()
