# 256-unit independent GRU agents

Status: prepared. This is a bounded 100-turn demonstration, not evidence of convergence.
Source: `7be9e420ddd3208102db71e34cfdee4e4c1145fd`, private Island Community main.
Configuration: `island_community/scenarios/neighbors_50_gru256.json`, seed 42,
50 founders, one independent GRU per island, vmap, CPU. The sole configuration
change relative to `neighbors_50.json` is hidden width 64 → 256. Parameters per
island: 751,470. Learning remains one-step actor–critic, Adam learning rate
0.0003, discount 0.99; history is eight explicit turns with gradients truncated
between turns. No death replay or new training objective is introduced.
The complete effective configuration is saved in the recording.

Authorized remote CPU: worker 0 of
`kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`,
project `he-vision-group`, zone `us-west4-a`, at most four threads. No TPU devices.
A fresh occupancy check is required immediately before launch.

Reuse the verified immutable source archive from the logits run:
SHA-256 `c74c84b40724bc97ffbe0a6e0233ef9f0b85f2abf452a43846a967b120ed526e`.
From `/tmp/island-community-logits-20261110-193900/source`:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/time -v -o /tmp/island-community-gru256-20261110-201500/timing.txt /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/scenarios/neighbors_50_gru256.json --turns 100 --device cpu --checkpoint-every 10 --output /tmp/island-community-gru256-20261110-201500/recording
```

Validation: 290 remote tests for the score recorder and replay changes, followed
by one focused remote 256-unit learning/checkpoint test, all passed.


## Completed simulation

All 100 turns completed with exit status 0, ending with one living island.
Wall time: 6:44.50; peak RSS: 2,511,260 KiB (2.395 GiB). This is observed total
CPU process memory, not measured GPU memory. The initial resident tensor estimate
of 430.04 MiB excludes transient computation and population-capacity growth.
The 64-unit run with logits took 4:00.35 with 749.00 MiB peak RSS. These are
single-seed demonstrations with different trajectories, not a controlled
throughput benchmark or evidence of convergence.

Artifact archive remote SHA-256:
`c63e9ad15b4520d2e0cae2da06c0bfdf57b28a1c981fd56ff618bef679512058`.
The archive contains the full model/optimizer checkpoint, effective configuration,
all frames and decision scores, journals, source hashes, timing, and checksums.


## Verified results and replay

Retrieved archive and every recorded file passed their SHA-256 checks; see
`artifact-verification.log`. There are 101 frames and 2,547
individual learning updates, seven births and 56 deaths. Of 442
trade proposals, 22 were accepted (4.98%);
20 trade contracts completed. Delivered cargo included 188 grain,
133 water, and 12 meals. The full event/resource summary is in `summary.json`.
All frames passed the degree-at-most-five, mutual-contact, and history-window
invariants. Producer access became incomplete in 22 frames as the population declined.
This single seeded run does not establish a benefit from the wider model.

The live read-only replay is http://127.0.0.1:39707/ . HTTP checks verified all
101 turns, the saved width 256, exactly 751,470 parameters per agent, initial
resident tensors of 450,933,600 bytes, and 291 action candidates for island 0
on turn 1. The same viewer passed the complete browser suite on both older and
newly scored 101-frame recordings; a further real-trade inspection covered all
220 quantity alternatives and the chosen row.

The automatic app-panel open request did not return promptly and was abandoned;
the localhost URL is available directly, and the replay server remains running.
All remote simulation and test processes completed. No TPU device was used.
