# Natural-language linked-facts dataset

This component generates a controlled communication problem for three agents:

```text
agent 1 input (one required hop) -- shared encoder -- message 1 --\
                                                                   receiver + question -> answer
agent 2 input (the other hop)    -- shared encoder -- message 2 --/
```

Each world mixes five relations. The middle is the private key that both
senders must mention. For each question, one sender receives only the
person-to-middle sentence and the other receives only the middle-to-answer
sentence. Which sender gets which hop is balanced across a world:

| Relation | Agent 1 | Agent 2 | Answer |
|---|---|---|---|
| membership | person belongs to a group | that group is based in a place | place |
| color | person has a color | that color is an animal's coat | animal |
| animal | person keeps an animal | that animal rests on furniture | furniture |
| furniture | person owns a piece of furniture | that piece stands in a room | room |
| pledged group | person is pledged to a house, order, clan, or similar group | that group gathers in a place | place |

The two sender agents use the same weights. Neither sees the question. The
receiver sees only the question and the two messages. Each sender therefore
communicates exactly one necessary link, with no unrelated facts in its input.
The model choices are written up in `MODEL_DESIGN.md`.

## World and question structure

Each JSONL row is one world containing one question per person. Every question
stores its own two single-fact sender inputs. The world-level metadata keeps the
complete set of latent chains for auditing without exposing unrelated chains to
the model.

```json
{
  "world_id": "train-00000000",
  "split": "train",
  "questions": [
    {
      "question_id": "train-00000000-q00",
      "domain": "color",
      "agent_1_input": "Rina's color is amber.",
      "agent_2_input": "The animal with an amber coat is the pelican.",
      "question": "Which animal has Rina's color?",
      "answer": "pelican",
      "answer_index": 0,
      "agent_1_template_id": "col-a1-f-00",
      "agent_2_template_id": "col-a2-f-00",
      "template_id": "col-q-f-00",
      "reasoning_path": ["Rina", "amber", "pelican"]
    }
  ],
  "metadata": {
    "entities": {},
    "chains": {},
    "domain_answers": {},
    "person_side_agent": {}
  }
}
```

Only `questions[].agent_1_input`, `questions[].agent_2_input`, and
`questions[].question` are model inputs. `questions[].answer` is the target.
Everything else is audit metadata and must be excluded from model inputs. An
optional training-only auxiliary objective may use the middle entry of
`questions[].reasoning_path` as a bridge label and
`metadata.person_side_agent` to identify which sender holds that
person-to-bridge fact. Neither label is passed to the sender or receiver as an
input.

## Evaluation splits

| Split | Entity-name bank | Sentence templates | Purpose |
|---|---|---|---|
| `validation` | familiar | familiar | model selection |
| `test_familiar` | familiar | familiar | new mappings |
| `test_new_names` | held out | familiar | lexical generalization |
| `test_new_wording` | familiar | held out | syntactic generalization |
| `test_new_names_and_wording` | held out | held out | combined generalization |

All names are invented. Familiar and held-out name banks are disjoint. The
held-out sentence templates are also disjoint, including question templates.
Each bank mixes different syntactic frames, such as passives, clefts,
existentials, appositives, fronted phrases, and complement clauses. A world
uses every frame in its bank before repeating one, so the eight facts in a
default world use eight different sentence structures. Sentence order still
varies independently of the underlying mappings.

Every world balances those relations and gives each one a random bijection. A
question exposes only the two facts on its own reasoning path. The correct
answer is computed by following:

```text
person -> middle -> answer
```

Neither sender alone has both hops of a chain, so neither can answer by itself.
The metadata makes this property easy to audit.

## Generate data

The default `starter` preset has 100,000 training questions, 5,000 validation
questions, and 5,000 questions in each of four test splits (125,000 total).

```bash
python -m linked_facts.generate_dataset \
  --output-dir datasets/linked_facts_v1 \
  --preset starter \
  --seed 1729
```

The `large` preset contains 1,250,000 total questions. Counts can be overridden
without editing code:

```bash
python -m linked_facts.generate_dataset \
  --output-dir datasets/linked_facts_custom \
  --preset large \
  --entities-per-world 12 \
  --train-worlds 200000 \
  --validation-worlds 10000 \
  --test-worlds-per-split 10000
```

Output is gzip-compressed JSONL by default. `manifest.json` records the full
configuration, exact counts, byte sizes, and an uncompressed SHA-256 digest for
each split. Generation is deterministic per `(seed, split, world index)`, so
changing one split's size does not change any existing worlds.

Verify every answer chain, count, stored byte size, and content checksum with:

```bash
python -m linked_facts.generate_dataset \
  --output-dir datasets/linked_facts_v1 \
  --verify-only
```

## Model-facing loader

`iter_question_examples` gives a simple flattened stream:

```python
from pathlib import Path

from linked_facts.generate_dataset import iter_question_examples

for example in iter_question_examples(Path("datasets/linked_facts_v1/train.jsonl.gz")):
    agent_1_text = example["agent_1_input"]
    agent_2_text = example["agent_2_input"]
    question = example["question"]
    target = example["answer"]
    bridge_target = example["bridge"]
    person_side_agent = example["person_side_agent"]
```

The model in `jax_models.py` is specified in `MODEL_DESIGN.md`. It uses one
Transformer for both senders and a second Transformer as receiver. The tiny
configuration uses one 32-dimensional continuous vector per sender and two
layers for both sender and receiver. The receiver reads both vectors plus the
character-tokenized question and scores up to 4 answer tokens. Passages are
characters, so a held-out spelling is visible to the sender. Each answer word
is one vocabulary item followed by an end token. The vocabulary includes every
name and lexicon string the generator can emit.

Training maximizes the joint log probability of those answer tokens:

```text
log p(answer | messages, question) = sum_t log p(answer_t | answer_<t, messages, question)
```

`JaxCommunicationModel` returns that sum for each example. Its `loss` is
the negative batch mean, for a standard minimizer. The same sender encodes both
private texts, so the answer objective trains both messages together.
