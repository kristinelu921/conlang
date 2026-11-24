"""Remote neural checks for constrained decoding, independent adapters, and frozen weights."""

import copy
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.neural


@pytest.fixture
def tiny_model_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small real Qwen architecture with the actual pinned tokenizer for fast gradient checks."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    def load(*arguments: Any, **keywords: Any) -> Qwen3ForCausalLM:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(128)
            config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48,
                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 head_dim=8, attention_dropout=0.0)
            config._attn_implementation = keywords["attn_implementation"]
            return Qwen3ForCausalLM(config)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)


@pytest.mark.parametrize("mode", ["frozen", "reward_adapter", "value_adapter"])
def test_decode_update_isolation_and_checkpoint(tiny_model_loader: None, mode: str, tmp_path: Path) -> None:
    """Check real constrained sampling, gradients, private history, and reproducible continuation."""
    import torch
    from island_community.config import load_config
    from island_community.qwen_backend import QwenPopulation
    from island_community.world import Action, Decision, TurnResult

    settings = load_config(Path(__file__).parents[1] / "scenarios" / "qwen_10_frozen.json")["language_model"]
    settings.update(mode=mode, max_reasoning_tokens=3, max_context_tokens=1024, adapter_rank=2)
    device = os.environ.get("ISLAND_QWEN_TEST_DEVICE", "cpu")
    population = QwenPopulation(settings, seed=42, observation_size=4, device=device)
    assert population.model.config._attn_implementation == population.runtime["attention_implementation"]
    population.add(0)
    population.add(1)
    population.histories[0] = [{"turn": 1, "reward": 2.0}]
    population._activate(0)
    choice, record, trajectory = population._choose(
        [{"role": "user", "content": 'Choose one: 0 wait, 1 request water, 2 request grain. Output {"choice":N}.'}], 3, 77,
    )
    assert record["raw_response"] == f'{{"choice":{choice}}}'
    assert 0 <= choice < 3
    assert trajectory["tokens"][-1] == population.tokenizer.eos_token_id
    original_checksum = population.base_checksum
    if mode == "frozen":
        assert not population.optimizers
        assert all(not parameter.requires_grad for parameter in population.model.parameters())
    else:
        other = [parameter.detach().clone() for module in population.adapter_modules
                 for parameter in (module.input_factors["1"], module.output_factors["1"])]
        population.pending = {0: [trajectory]}
        population.last_decision_scores = {0: {"action": record}}
        decision = Decision([0.2, 0.4, -0.3, 0.1], [Action("wait")], [[0.0]])
        outcome = TurnResult({0: 2.0}, {}, [], {0: 1.0})
        metrics = population.learn({0: decision}, outcome, {0: [0.1, 0.3, -0.2, 0.0]})
        assert metrics["updates"] == 1
        learning = population.last_decision_scores[0]["learning"]
        assert learning["adapter_change_l2"] > 0
        assert all(torch.equal(before, after) for before, after in zip(other,
            [parameter for module in population.adapter_modules for parameter in (module.input_factors["1"], module.output_factors["1"])]))
        assert population.updates == {0: 1, 1: 0}
        if mode == "value_adapter":
            assert learning["target_scaled"] == pytest.approx(0.2 + settings["discount"] * learning["next_value_scaled"])
        population.add(2, parent_id=0)
        assert population.histories[2] == [] and population.updates[2] == 0
        assert not population.optimizers[2].state
        for module in population.adapter_modules:
            assert torch.equal(module.output_factors["0"], module.output_factors["2"])
            assert module.output_factors["0"].data_ptr() != module.output_factors["2"].data_ptr()
    assert population.frozen_checksum() == original_checksum
    checkpoint_path = tmp_path / "policies.pt"
    torch.save(population.state_dict(), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    restored = QwenPopulation.from_state_dict(checkpoint, device=device)
    assert restored.histories == population.histories
    assert restored.updates == population.updates
    population._activate(0)
    restored._activate(0)
    messages = [{"role": "user", "content": 'Select 0 or 1. Output {"choice":N}.'}]
    expected = population._choose(messages, 2, 912)
    actual = restored._choose(messages, 2, 912)
    assert expected[0] == actual[0]
    assert expected[2] == actual[2]


def test_recorded_sampling_support_survives_changed_replay_ranking(tiny_model_loader: None) -> None:
    """Keep the sampled distribution's support when replay changes an excluded token's rank."""
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from island_community.config import load_config
    from island_community.qwen_backend import QwenPopulation

    settings = load_config(Path(__file__).parents[1] / "scenarios" / "qwen_10_frozen.json")["language_model"]
    settings.update(max_reasoning_tokens=3, max_context_tokens=1024)
    population = QwenPopulation(settings, seed=42, observation_size=4,
                                device=os.environ.get("ISLAND_QWEN_TEST_DEVICE", "cpu"))
    _, _, trajectory = population._choose([{"role":"user", "content":'Select 0 or 1. Output {"choice":N}.'}], 2, 77)
    supported = set().union(*trajectory["reasoning_allowed"], *trajectory["answer_allowed"])
    outside = next(token for token in range(population.model.config.vocab_size) if token not in supported)
    expected = population._trajectory_log_probability(trajectory)

    def change_excluded_score(module: torch.nn.Module, arguments: tuple[Any, ...],
                              output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        """Make an unselected, unsupported token dominate a recomputed truncation filter."""
        output.logits = output.logits.clone()
        output.logits[..., outside] += 1000
        return output

    handle = population.model.register_forward_hook(change_excluded_score)
    try:
        actual = population._trajectory_log_probability(trajectory)
    finally:
        handle.remove()
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(os.environ.get("ISLAND_QWEN_TEST_DEVICE") != "tpu", reason="Native TPU shape regression")
def test_tpu_chunked_prefill_matches_cpu_for_qwen_attention_shape() -> None:
    """Compare full-size attention heads and chunked cached decoding with a CPU reference."""
    import torch
    import torch_tpu
    from transformers import AttentionInterface, Qwen3Config, Qwen3ForCausalLM, StaticCache
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    from island_community.qwen_backend import chunked_qwen_attention

    torch.tpu.distributed.set_tpu_launch_env(nnodes=1, nproc_per_node=1)
    AttentionInterface.register("island_chunked_eager", chunked_qwen_attention)
    AttentionMaskInterface.register("island_chunked_eager", eager_mask)
    torch.manual_seed(128)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=8,
                         head_dim=128, attention_dropout=0.0, eos_token_id=127, pad_token_id=127)
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model = copy.deepcopy(reference).to("tpu")
    model.set_attn_implementation("island_chunked_eager")
    tokens = (torch.arange(600).reshape(1, -1) % 100).long()
    attention_mask = torch.ones_like(tokens)
    with torch.inference_mode():
        expected_logits = reference(tokens, attention_mask=attention_mask, use_cache=False).logits[:, -1]
        actual_logits = model(tokens.to("tpu"), attention_mask=attention_mask.to("tpu"),
                              use_cache=False).logits[:, -1].cpu()
        torch.testing.assert_close(actual_logits, expected_logits, rtol=5e-3, atol=5e-3)
        arguments = dict(max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=127)
        expected = reference.generate(tokens, attention_mask=attention_mask, **arguments)
        actual = model.generate(tokens, attention_mask=attention_mask, **arguments,
                                past_key_values=StaticCache(config=config, max_cache_len=1024),
                                prefill_chunk_size=512, disable_compile=True)
    assert torch.equal(actual.cpu(), expected)


def test_chunked_attention_preserves_causal_outputs_and_all_input_gradients() -> None:
    """Isolate chunking's output and gradient effects from ordinary CPU/TPU numerical differences."""
    import json
    import torch
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    from island_community.qwen_backend import chunked_qwen_attention

    device = os.environ.get("ISLAND_QWEN_TEST_DEVICE", "cpu")
    if device == "tpu":
        import torch_tpu

        torch.tpu.distributed.set_tpu_launch_env(nnodes=1, nproc_per_node=1)
    module = torch.nn.Module().eval()
    module.num_key_value_groups = 2
    generator = torch.Generator().manual_seed(128)
    shapes = ((1, 16, 577, 128), (1, 8, 577, 128), (1, 8, 577, 128))
    inputs = [torch.randn(shape, generator=generator, requires_grad=True) for shape in shapes]
    remote_inputs = [tensor.detach().to(device).clone().requires_grad_() for tensor in inputs]
    eager_inputs = [tensor.detach().to(device).clone().requires_grad_() for tensor in inputs]
    chunked_cpu_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    mask = torch.full((1, 1, 577, 577), torch.finfo(torch.float32).min).triu(diagonal=1)
    expected = eager_attention_forward(module, *inputs, mask, scaling=128 ** -0.5)[0]
    eager_remote = eager_attention_forward(module, *eager_inputs, mask.to(device), scaling=128 ** -0.5)[0]
    chunked_cpu = chunked_qwen_attention(module, *chunked_cpu_inputs, mask, scaling=128 ** -0.5)[0]
    actual = chunked_qwen_attention(module, *remote_inputs, mask.to(device), scaling=128 ** -0.5)[0]
    weights = torch.randn(expected.shape, generator=generator)
    (expected * weights).sum().backward()
    (eager_remote * weights.to(device)).sum().backward()
    (chunked_cpu * weights).sum().backward()
    (actual * weights.to(device)).sum().backward()
    comparisons = [("output", expected.detach(), eager_remote.detach().cpu(), actual.detach().cpu())]
    for index, (reference, cpu_result, eager_result, result) in enumerate(zip(inputs, chunked_cpu_inputs, eager_inputs, remote_inputs)):
        assert result.grad is not None
        comparisons.append((f"gradient_{index}", reference.grad, eager_result.grad.cpu(), result.grad.cpu()))
    for label, reference, baseline, chunked in comparisons:
        print(json.dumps({"comparison": label,
                          "baseline_cpu_tpu_max_absolute": float((baseline - reference).abs().max()),
                          "baseline_cpu_tpu_relative_l2": float((baseline - reference).norm() / reference.norm()),
                          "chunked_eager_max_absolute": float((chunked - baseline).abs().max()),
                          "chunked_eager_relative_l2": float((chunked - baseline).norm() / baseline.norm())}))
    torch.testing.assert_close(chunked_cpu.detach(), expected.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual.detach().cpu(), eager_remote.detach().cpu(), rtol=5e-3, atol=5e-3)
    for reference, cpu_result, eager_result, result in zip(inputs, chunked_cpu_inputs, eager_inputs, remote_inputs):
        torch.testing.assert_close(cpu_result.grad, reference.grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result.grad.cpu(), eager_result.grad.cpu(), rtol=5e-3, atol=5e-3)


def test_qwen_runner_preserves_exact_actions_and_resume(tiny_model_loader: None, tmp_path: Path) -> None:
    """Run the full recorder/checkpoint path with actual constrained Qwen decisions."""
    import json
    from island_community.config import load_config
    from island_community.runner import run
    from island_community.storage import load_checkpoint

    configuration = load_config(Path(__file__).parents[1] / "scenarios" / "qwen_10_frozen.json")
    configuration["language_model"]["max_reasoning_tokens"] = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration))
    device = os.environ.get("ISLAND_QWEN_TEST_DEVICE", "cpu")
    first = run(tmp_path / "first", turns=1, config_path=config_path, checkpoint_every=1, device=device)
    second = run(tmp_path / "second", turns=1, resume=first / "checkpoint.pt", checkpoint_every=1, device=device)
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["policy"] == "qwen_frozen"
    assert manifest["language_model"]["base_sha256_before"] == manifest["language_model"]["base_sha256_after"]
    checkpoint = load_checkpoint(second / "checkpoint.pt")
    assert all(value == 0 for value in checkpoint["policies"]["updates"].values())
    for directory in (first, second):
        records = [json.loads(line) for line in (directory / "experiences.jsonl").read_text().splitlines()]
        for record in records:
            saved = json.loads((directory / "decisions" / f'{record["turn"]}.json').read_text())["islands"][str(record["island_id"])]
            assert saved["execution"]["requested_action"] == saved["execution"]["submitted_action"] == record["action"]
            assert saved["execution"]["action"] == record["action_results"][0]["action"]
            context = json.loads(saved["action"]["messages"][1]["content"])["your_state"]
            assert len(context["history"]) == record["turn"] - 1
