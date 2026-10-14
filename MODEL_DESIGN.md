# Communication model design

This note records the choices in `models.py` and the reasons for them. The
task is the linked-facts game: one shared sender reads each private passage,
and a receiver (model C) must answer a question from the two messages alone.
`ControlModel` in `control_model.py` is the single model that sees both
passages and the question. `train.py` trains the communication model, and
`control_train.py` trains the control model.

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
| Initialization | PyTorch defaults | You asked for standard initializations. Nothing is re-sampled after construction. |
| Control model | Encoder-decoder in `control_model.py`, hidden 192, 6 heads, 2 layers, feedforward 768 | Same answer objective with no message bottleneck. Width 192 is about twice the parameter count of the 128-wide model, and 6 heads keep each head 32 wide. |
| Optimizer | AdamW, learning rate `3e-4`, gradient clip `1.0` | AdamW is the usual Transformer optimizer. `3e-4` is a standard small-model step size. The clip limits a step from the large initial embeddings. |
| Weight decay | `0.01` on matrices, `0` on biases and layer-norm scales | `0.01` is AdamW's usual default. Vector parameters such as layer-norm scales are left without decay so the decay does not shrink them toward zero. |

Vocabulary size is 2191. Parameter counts, with the shared word embedding listed once:

| Piece | Parameters |
|---|---|
| Word embedding shared by the sender and the receiver | 280,448 |
| Sender, apart from that embedding | 463,104 |
| Receiver, apart from that embedding | 1,226,255 |
| Communication model, embedding counted once | 1,969,807 |
| Control transformer, including its own embedding | 3,117,583 |

Agent 1 and agent 2 are the same sender, so they are the same size. The receiver is larger than the sender because it has both an encoder and a decoder. The control model has 3,117,583 parameters, 1.92 times a 128-wide encoder-decoder of the same depth. Hidden size 256 would be 5,073,551 parameters, because the attention and feedforward matrices grow with the square of the width.

## Architecture

Both agents call `Sender`. It prepends 4 learned query embeddings to the
character embeddings, adds learned absolute positions, and runs a
batch-first Transformer encoder. The returned message is the encoder state at
the 4 query positions, shape `(batch, 4, 128)`. `encode_messages` runs that
same module twice and concatenates on the length axis, shape `(batch, 8, 128)`.

`Receiver` is an encoder-decoder:

1. The 8 message vectors are placed in positions 0–7. They are not passed
   through the token embedding, because they are already hidden states.
2. The question is embedded with the shared token table and placed in the
   following positions.
3. A 2-layer encoder mixes the messages with the question. Its output is the
   memory the decoder cross-attends to.
4. A 2-layer decoder predicts the answer under a causal mask. The output head
   is a separate linear map from 128 to the vocabulary.

Layer defaults are the PyTorch 2.14 `TransformerEncoderLayer` and
`TransformerDecoderLayer` defaults, aside from the sizes above: ReLU, post-norm,
layer-norm epsilon `1e-5`, bias on, dropout 0.1. Post-norm and ReLU are the
library standard for this block. Pre-norm and GELU were not substituted.

Position tables are learned embeddings, not sinusoids. The sender table covers
4 message slots plus 512 text characters. The receiver encoder table covers 8
message slots plus 128 question characters. The answer position table has 4
rows. Measured passages are a few hundred characters and questions are under
80, so these caps hold the current generator without truncating.

The encoder is built with `enable_nested_tensor=False`, so padding uses an
explicit key-padding mask instead of the nested-tensor fast path.

## KV structure

Inside each attention block, query, key, and value share one projection
(`in_proj_weight`, shape `(3 * 128, 128)`), initialized with Xavier uniform.
The output projection is a separate linear layer. There is no extra KV bias
and no zero-attention slot. Four heads each see 32-dimensional keys and values.

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
word and the end token. Padding positions use `ignore_index` and add nothing.

```text
log p(answer | messages, question) = sum_t log p(answer_t | answer_<t, messages, question)
loss = -mean_batch of that sum
```

The sum is the joint probability of the answer string. A mean over tokens would
treat a short answer and a long one as the same number of decisions, which is a
different objective. `CommunicationOutput.joint_log_probability` is the per-row
sum. `loss` is what a minimizer should step on. The control transformer uses the
same sum, with the full character context in place of the two messages.

## Initialization

No custom reset runs after module construction.

| Parameter | Scheme |
|---|---|
| Token, position, and query embeddings | Normal with mean 0 and standard deviation 1. The padding row of the token embedding stays zero. |
| Attention input projection | Xavier uniform. Its bias is zero. |
| Attention output projection and the vocabulary projection | Kaiming uniform on the weights, which is PyTorch's linear default. Attention output bias is then set to zero by the attention module. The vocabulary bias uses the linear default. |
| Feedforward linears | Kaiming uniform, the linear default. |
| Layer norm | Weight 1, bias 0. |

## Control transformer

`ControlModel` in `control_model.py` is one encoder-decoder with hidden size 192, 6 heads, 2
layers, and feedforward width 768. Dropout and initialization match the other
models. Its input is the character
sequence `agent 1 passage + agent 2 passage + question`, capped at 1024
positions. Measured full contexts reach 944 characters. It has no query slots
and no message bottleneck. The answer head is the same 4-token decoder, and the
loss is the same joint log probability.

## Training

`train.py` trains the communication model. `control_train.py` trains the control model on the same loss and the same optimizer settings. Both read a generated dataset directory and optimize that loss with
AdamW. Matrices, including the `N(0, 1)` embeddings, use weight decay `0.01`.
Biases and layer-norm scales use weight decay `0`. The learning rate is
`3e-4`, the gradient clip is `1.0`, and the batch size is 32. There is no
learning-rate schedule. Evaluation runs the model in eval mode, so dropout is
off, and reports greedy exact match. `greedy_answer_token_ids` itself does not
switch modes.

Both trainers log to the Weights & Biases project `conlangs` unless
`--disable-wandb` is set. Each step records the loss, the answer-token
distribution, global and per-module weight and gradient norms, and activation
RMS. For the communication model the three gradient series are the agent-1
sender path, the agent-2 sender path, and the receiver, with the shared
embedding split across those paths. Each series also logs the value distribution
of its four intermediate vectors.

On a TPU, omit `--dataset-dir`. The trainers read the instance zone, select the one
`kmh-gcp` bucket whose location is that region, and stage
`kristine/datasets/conlangs` from that bucket. A zone in another region never
falls back to the us-west4 copy. A local directory is only used when
`--dataset-dir` is passed.

```bash
python train.py --dataset-dir datasets/linked_facts_v1
python control_train.py --dataset-dir datasets/linked_facts_v1
python train.py
python control_train.py
```
