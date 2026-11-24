# Native TPU check 2 — passed

All four Qwen backend tests passed on the same authorized v5e-4 in `us-west4-a`: **4 passed, 1 experimental API warning, 366.11 seconds**, exit sentinel `0`. The exact command and environment are in `native-check2.sh`, and the complete output is in `native-check2.log`. The source archive and pinned package versions match check 1.

The tests cover constrained JSON generation, frozen-weight hashes, independent reward-trained adapters, independent adapters with a TD value baseline, inheritance with fresh history and optimizer state, deterministic checkpoint restoration, and a full ten-island turn followed by checkpoint resume into a second turn. Each recorded requested/submitted/world action matches, and each prompt contains the expected number of prior private turns. Tiny randomly initialized Qwen weights are used, with the real pinned tokenizer; these results establish backend correctness, not pretrained policy quality.

Task-local `TPU_LOG_DIR` and `GOOGLE_LOG_DIR` eliminated the previous shared-directory permission errors. No shared permissions were changed. The completed process left another empty current-user lock timestamped `2026-11-11 03:04:12.749814697 +0000`; the user has now explicitly authorized cleaning up our test locks after proving no TPU device or lock is held.

The pinned public `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` was downloaded directly to this VM's isolated Hugging Face cache; `download-pinned-model.sh` and its output record the command and node-local snapshot path. No dataset was transferred across zones or regions.

During this test, the development machine's root filesystem filled. Our inactive, current-user-owned `/tmp/island-community-install-check` environment was relocated intact to `local-staging/install-check` under this report, preserving its original path with a symlink and freeing 24 MB. The remote run was unaffected; no experiment recordings, checkpoints, or other users' files were removed.
