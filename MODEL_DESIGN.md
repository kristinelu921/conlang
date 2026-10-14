# Communication model design

This note records the choices in `jax_models.py` and the reasons for them. The
task is the linked-facts game: one shared sender reads each private passage,
and a receiver must answer a question from the two messages alone.

## Choices

| Choice | Value | Why |
|---|---|---|
| Sender | One Transformer encoder, shared by both agents | You asked for one small model that both agents use. Sharing forces the same code to carry person-to-middle facts and middle-to-answer facts. |
| Message | 4 vectors, each of size 128, per agent | You asked for 4 tokens and a hidden size of 128. The 4 positions are learned query slots read out of the encoder, so the channel length is fixed even when the passage is hundreds of characters. |
| Receiver input | Those 8 vectors, then the question | The receiver never sees either passage. Concatenating the messages in agent order keeps which sender said what. |
| Receiver output | Up to 4 answer tokens | You asked for an answer of at most 4 tokens. Decoding stops at the end token. |
| Width and depth | Hidden 128, 2 layers, 4 heads, feedforward 512 | 128 was requested. Two layers and four heads keep it a small encoder-decoder. Feedforward width is the usual 4 times the hidden size from Attention Is All You Need. Each head is 32 wide. |
| Attention KV | Standard multi-head attention | One fused query/key/value projection per block, full keys and values for every head. No grouped-query or multi-query attention, and no KV cache. |
| Channel contents | Continuous hidden states | The joint answer log probability has to train both messages. Continuous vectors let that gradient pass through the sender. A discrete codebook would need an extra sampling step that was not requested. |
| Tokenizer | Characters for text, one token per answer word | Passages must expose held-out spellings. Answers are 4–13 characters, so a character answer cannot fit in 4 tokens. |
| Objective | Sum of answer-token log probabilities | You asked for joint log probabilities. The scalar loss is the negative of that sum, averaged over the batch. |
| Initialization | Flax linen defaults, with token embeddings drawn from a normal of standard deviation 1 | Standard library initialization. The embedding scale is the one explicit choice. |
| Optimizer | AdamW, learning rate `3e-4`, gradient clip `1.0` | AdamW is the usual Transformer optimizer. `3e-4` is a standard small-model step size. The clip limits a step from the large initial embeddings. |
| Weight decay | `0.01` on matrices, `0` on biases and layer-norm scales | `0.01` is AdamW's usual default. Vector parameters such as layer-norm scales are left without decay so the decay does not shrink them toward zero. |
| Position encoding | T5 relative attention bias, 32 buckets, maximum distance 128 | Relative buckets preserve order without allocating absolute tables at every supported sequence length. |

Vocabulary size is 2191. Parameter counts, with the shared word embedding listed once:

| Piece | Parameters |
|---|---|
| Word embedding shared by the sender and the receiver | 280,448 |
| Sender, apart from that embedding | 397,312 |
| Receiver, apart from that embedding | 1,209,103 |
| Communication model, embedding counted once | 1,886,863 |

Agent 1 and agent 2 are the same sender, so they are the same size. The receiver
is larger than the sender because it has both an encoder and a decoder. These
counts come from the Flax parameter tree for the configuration above.

## Architecture

Both agents call `Sender`. It prepends 4 learned query embeddings to the
character embeddings and runs a batch-first Transformer encoder. The returned
message is the encoder state at the 4 query positions, shape `(batch, 4, 128)`.
`encode_messages` runs that module twice and concatenates on the length axis,
shape `(batch, 8, 128)`.

`Receiver` is an encoder-decoder:

1. The 8 message vectors are concatenated ahead of the question. They are not
   passed through the token embedding, because they are already hidden states.
2. The question is embedded with the shared token table and placed after the
   messages.
3. A 2-layer encoder mixes the messages with the question. Its output is the
   memory the decoder cross-attends to.
4. A 2-layer decoder predicts the answer under a causal mask. The output head
   is a separate linear map from 128 to the vocabulary.

Each block is post-norm with ReLU and dropout 0.1, built from Flax
`MultiHeadDotProductAttention` and `Dense`. Attention softmax runs in float32.
`deterministic` is passed on every call, so evaluation disables dropout even
after a training call has built the module. Pre-norm and GELU were not substituted.

Sequence order comes from a T5 relative attention bias, not from absolute
position embeddings. Each head has 32 learned buckets. Offsets up to half of
the buckets are exact, and longer offsets share logarithmic buckets out to
distance 128. Encoder self-attention and decoder cross-attention are
bidirectional. Decoder self-attention only buckets distances into the past.
The 4 sender query slots stay learned content vectors; their order relative
to the passage comes from this bias. Measured passages are a few hundred
characters and questions are under 80, so the configured caps of 512 and 128
hold the current generator without truncating.

Padding uses an explicit boolean attention mask.

## KV structure

Inside each attention block, query, key, and value are produced for every head
and the heads are mixed by an output projection. There is no grouped-query or
multi-query attention and no KV cache. Four heads each see 32-dimensional keys
and values.

The communication channel is not a token KV cache. It is the 8 continuous
vectors. After the receiver encoder, those vectors and the question tokens
become the decoder's cross-attention memory: every decoder step can attend to
all 8 message positions and every question character. Greedy generation reruns
the decoder on the growing prefix. With at most 4 answer steps, a cache would
not change the result.

Padding masks are boolean. Message positions are never masked. Text, question,
and answer pads are masked. The answer causal mask is also boolean, with future
positions marked true.

## Tokenizer

`TextTokenizer` builds one closed vocabulary:

- specials, in order: `<pad>` 0, `<bos>` 1, `<eos>` 2, `<unk>` 3
- 75 characters: ASCII letters and digits, plus space and `.,;:?'"!-()/`
- 2112 answer words: all 1920 generated names and all 192 lexicon strings,
  familiar and held-out

`encode_text` maps a passage or question one character at a time, using `<unk>`
for anything outside that character set. `encode_answer` maps the whole answer
string to one id and appends `<eos>`, so a normal target has length 2 and stays
inside the 4-token limit. Held-out names are in the vocabulary, so the receiver
can emit a spelling it has never been trained to produce. It does not assemble
that spelling from characters.

The token embedding is shared by the sender and the receiver. The answer output
projection is not tied to it.

## Objective

Teacher forcing shifts the target right and starts the decoder at `<bos>`. For
an answer encoded as `[word, eos]` and padded to 4, the scored tokens are the
word and the end token. Padding positions are masked out of the sum.

```text
log p(answer | messages, question) = sum_t log p(answer_t | answer_<t, messages, question)
loss = -mean_batch of that sum
```

The sum is the joint probability of the answer string. A mean over tokens would
treat a short answer and a long one as the same number of decisions, which is a
different objective. `CommunicationOutput.joint_log_probability` is the per-row
sum. `loss` is what a minimizer should step on.

## Initialization

Token embeddings, the sender query slots, and the relative-position bucket
tables use a normal initializer with standard deviation 1. The other
projections and layer norms use Flax linen defaults. Nothing is re-sampled
after the module is built.

## Training

`train.py` trains the communication model with JAX, Flax, and Optax. It pads
every passage, question, and answer to one configured static shape, tokenizes
the dataset once on the host, shards each global batch across all local TPU
devices with `jax.pmap`, and averages gradients across those devices. This
avoids shape-driven recompilation and per-step Python tokenization on TPU.

The JAX path uses AdamW with learning rate `3e-4`, matrix-only weight decay
`0.01`, gradient clipping at `1.0`, and no learning-rate schedule. The TPU gate
uses bfloat16 model compute with float32 log-softmax and loss calculations.
Evaluation disables dropout and reports greedy exact match.

The optimized trainer logs loss, global gradient norm, and steady-state
examples per second at a configurable interval. Throughput accounting excludes
compilation and warmup. On TPU, omitting `dataset_dir` makes the trainer verify
the TPU zone, require the matching regional bucket, and use the configured
node-local cache under `/home/<user>/elf-data/`.

```bash
JAX_PLATFORMS=cpu python train.py --dataset-dir datasets/linked_facts_v1 --disable-wandb
JAX_PLATFORMS=tpu python train.py --config configs/conlang-jax-v5e8-gate1.yaml
```
