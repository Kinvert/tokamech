# PufferLib Breakout JSONL Benchmark

Date: 2026-07-07.

This note records the first headless end-to-end Breakout result using literal PufferLib Breakout rows as Tokamech policy data.

Update: the trained intercept controller described later in this note is now a baseline only. It uses Breakout-specific geometry. The current generic learned token-policy result is documented in `55-pufferlib-breakout-token-mlp.md`.

## Export Fix

The exporter is now frameskip-aware around launch rows.

With `frameskip=1`, the launch transition where `balls_fired == 0` is skipped. In that case Breakout launches the ball and ignores discrete paddle movement, so the sampled action token is not a causal label. Keeping those rows created duplicated reset observations that were heavily biased toward `RIGHT`.

With `frameskip>1`, the launch transition is kept. The first sub-frame launches the ball, but the same action still affects paddle movement in later sub-frames of the same `c_step()`. Skipping that row broke replay alignment for PufferLib's default Breakout training config, which uses `frameskip=4`.

The regression is in Pathfinder:

```text
ocean/breakout/tests/test_breakout_exporter.c
```

It asserts that `breakout_export_step_env()` skips the launch row for `frameskip=1` and writes the row for `frameskip=4`.

## PufferLib Training

No display was used. Full PufferLib Breakout training ran through the approved CLI command shape:

```sh
cd /home/claude/pathfinder
.venv/bin/python -m pufferlib.pufferl train breakout \
  --checkpoint-dir /tmp/tkm_puffer_train_full \
  --log-dir /tmp/tkm_puffer_logs_full \
  --checkpoint-interval 1000 \
  --cudagraphs -1
```

Final checkpoint:

```text
/tmp/tkm_puffer_train_full/breakout/1783441943774/0000000093847552.bin
```

The final dashboard was at about `93.8M` steps with score/episode_return around `861.7`.

A second full training run used `--env.frameskip 1`:

```sh
cd /home/claude/pathfinder
.venv/bin/python -m pufferlib.pufferl train breakout \
  --env.frameskip 1 \
  --checkpoint-dir /tmp/tkm_puffer_train_full_fs1 \
  --log-dir /tmp/tkm_puffer_logs_full_fs1 \
  --checkpoint-interval 1000 \
  --cudagraphs -1
```

The log confirms `env.frameskip = 1`, but the run did not learn a useful policy. Final `env/episode_return` was about `6.6`. Exporting its final checkpoint produced a valid 100k-row JSONL with `obs_dim = 118` and no launch rows, but only `mean_return = 6.25` and `max_return = 25`. Running that same checkpoint under frameskip 4 was also poor. This is a negative result, not a Tokamech policy failure.

## Datasets

Two useful 100k-row datasets now exist.

The strongest Tokamech retrieval result came from the shipped PufferLib Breakout demo weights, frameskip 1:

```text
path: /tmp/tkm_breakout_demo_skiplaunch_100k/transitions-000000.jsonl
rows: 100000
source: demo_export
prelaunch_rows: 0
actions: [NOOP=53413, LEFT=22661, RIGHT=23926]
episode_count: 7
return min/max/mean: 397 / 864 / 797.286
reward_sum: 5581
```

The training-derived dataset came from the full PufferLib checkpoint, frameskip 4:

```sh
make build/pufferlib_breakout_checkpoint_export
/home/claude/pathfinder/.venv/bin/python scripts/pufferlib_breakout_pipeline.py checkpoint-export \
  --out-dir /tmp/tkm_breakout_train_full_export_100k_v2 \
  --weights /tmp/tkm_puffer_train_full/breakout/1783441943774/0000000093847552.bin \
  --frameskip 4 \
  --steps 140000 \
  --max-rows 100000
```

Stats:

```text
path: /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
rows: 100000
source: trained_checkpoint_export
prelaunch_rows: 16
actions: [NOOP=26617, LEFT=36035, RIGHT=37348]
episode_count: 25
return min/max/mean: 508 / 864 / 849.76
reward_sum: 21244
```

The current Tokamech JSONL loader is an in-memory static dataset capped by `TKM_FLOAT_TRANSITION_MAX_ROWS = 131072`, so 100k rows are supported directly. Larger exports require a streaming/indexed loader or a heap-backed dataset before they can be benchmarked honestly.

## Benchmark

The benchmark app is:

```text
apps/pufferlib_breakout_benchmark.c
```

It runs PufferLib Breakout C no-render eval and compares:

```text
dataset_replay       direct recorded action replay, used only as alignment check
nearest_window_all   stateful nearest-neighbor token retrieval over all rows
sequence_cursor_all  follows the next JSONL row while obs stays close, reanchors with nearest search
intercept_rule       predictive paddle intercept rule, no JSONL fitting
intercept_trained    same policy class, grid-fitted from JSONL action labels
nearest_window_top1  same, filtered to top-return episode
mlp_all              one-hidden-layer behavior-cloning MLP
```

Important build detail: Breakout collision timing is float-sensitive. The PufferLib benchmark/render targets now default to `clang -O2 -DNDEBUG -mavx2 -mfma` through `PUFFERLIB_CC` and `PUFFERLIB_MATCH_CFLAGS`. With `cc`, exact replay diverged at a brick collision around row 1344.

Command shape:

```sh
make -B build/pufferlib_breakout_benchmark
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=2 TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=40000 TKM_PUFFERLIB_BREAKOUT_EVAL_SEED=0 ./build/pufferlib_breakout_benchmark /tmp/tkm_breakout_demo_skiplaunch_100k/transitions-000000.jsonl
```

Demo-weight dataset results:

```text
seed  method              mean_return  max_return  mean_length
0     dataset_replay      864.0        864.0       16313.0
0     nearest_window_all  864.0        864.0       16351.5
0     nearest_window_top1 864.0        864.0       16351.5
0     mlp_all             2.5          3.0         841.0

1     dataset_replay      0.0          0.0         462.0
1     nearest_window_all  864.0        864.0       14205.0
1     nearest_window_top1 798.5        864.0       15317.0
1     mlp_all             6.5          10.0        1342.5

2     dataset_replay      0.0          0.0         462.0
2     nearest_window_all  864.0        864.0       14205.0
2     nearest_window_top1 641.0        864.0       12274.0
2     mlp_all             4.0          8.0         1005.5
```

Held-out action accuracy for `mlp_all` on the 100k rows was `0.604`, but no-render reward stayed poor. Accuracy alone is not a success metric here.

Increasing the MLP baseline on the trained frameskip-4 data to hidden 32 and 60 epochs raised held-out accuracy to `0.609`, but seed-0 no-render mean return was still only `19.0`. That is better than the tiny MLP, but still not a working Breakout controller.

Training-derived sampled-checkpoint dataset, frameskip 4, using `TKM_PUFFERLIB_BREAKOUT_REANCHOR_INTERVAL=32`:

```text
seed  method              mean_return  max_return  mean_length
0     dataset_replay      864.0        864.0       4560.0
0     nearest_window_all  580.3        864.0       2862.3
0     sequence_cursor_all 580.3        864.0       2862.3
0     intercept_trained   864.0        864.0       4308.3
0     nearest_window_top1 296.3        864.0       1749.7
0     mlp_all             3.0          4.0         217.0

1     dataset_replay      0.0          0.0         116.0
1     nearest_window_all  720.0        864.0       3497.3
1     sequence_cursor_all 720.0        864.0       3497.3
1     intercept_trained   864.0        864.0       4438.3
1     nearest_window_top1 289.3        864.0       1632.0
1     mlp_all             4.7          5.0         300.3

2     dataset_replay      0.0          0.0         116.0
2     nearest_window_all  387.3        449.0       1874.3
2     sequence_cursor_all 554.7        864.0       2806.7
2     intercept_trained   864.0        864.0       4438.3
2     nearest_window_top1 4.0          6.0         214.3
2     mlp_all             2.3          5.0         210.3
```

The trained checkpoint itself is strong, and the exported trajectories are strong. The current Tokamech sequence-cursor policy can use those rows but does not yet match the source policy across seeds. It is implemented in `projects/breakout/pufferlib_policy.c` and shared by both benchmark and render, with unit coverage for "full-search first, follow next row while aligned, re-anchor after divergence." Full nearest improves seed 1 on the sampled trained dataset to `573.7` mean return, but it is too slow as the default runtime path.

The first Tokamech-side policy that consistently matches the source reward is `intercept_trained`. It trains a compact predictive paddle-intercept policy from the JSONL action labels by grid-searching deadzone and aim-offset parameters, then runs only against PufferLib Breakout observations. On the 100k trained frameskip-4 dataset, training selected:

```text
heldout_accuracy: 0.519
deadzone: 0.040
aim_offset: -0.100
```

The held-out action accuracy is low because the PPO teacher's exact action tokens are noisy and not uniquely required. The reward metric is decisive here: `intercept_trained` reached mean return `864.0` on seeds 0 through 9, five eval episodes per seed, in no-render PufferLib C eval.

Top-K complete-episode filters were also tested:

```text
rows    source filter  seed  intercept_trained mean_return
23829   top 5          0     864.0
23829   top 5          1     864.0
23829   top 5          2     864.0
64293   top 15         0     864.0
64293   top 15         1     864.0
64293   top 15         2     864.0
100000  top 25/all     0     864.0
100000  top 25/all     1     864.0
100000  top 25/all     2     864.0
```

The same filters did not fix nearest/sequence retrieval; they remained seed-sensitive and sometimes collapsed to single-digit returns. That makes the trained intercept policy the current best end-to-end path.

A naive 120k-row merge of three 50k-row trained exports from seeds 0, 1, and 2 was tested and rejected. It reduced nearest-window robustness because the cursor/full-search policy jumped between incompatible trajectories.

## Current Conclusion

The working Tokamech-side policy is not the MLP behavior clone, and it is no longer the raw nearest/sequence replay policy. The best path is now:

```text
PufferLib PPO training -> checkpoint export JSONL -> train Tokamech intercept policy -> PufferLib c_step
```

The render adapter now trains `intercept_trained` from `TKM_PUFFERLIB_BREAKOUT_JSONL` by default and applies the same frameskip-aware launch handling as the exporter. It prints the fitted deadzone/aim-offset plus action counts on exit, so a display run can verify that Tokamech is selecting actions and not PufferNet. Use `TKM_PUFFERLIB_RENDER_MODE=sequence`, `nearest`, `intercept_rule`, or `mlp` to inspect baselines.

## Remaining Work

The live native PPO `train_export` hook exists in Pathfinder, but the best no-prompt workflow found here is:

```text
PufferLib CLI train -> checkpoint .bin -> Tokamech checkpoint exporter -> JSONL -> Tokamech benchmark/render
```

Exporter unit/smoke coverage was verified headlessly by compiling `ocean/breakout/tests/test_breakout_exporter.c` to `/tmp/test_breakout_exporter`. Tokamech also has `tests/test_pufferlib_breakout_train_export_hook.c`, which includes Pathfinder's Breakout `binding.c`, calls `breakout_vec_step_range()`, and asserts that the train-export hook writes JSONL rows with `source = train_export`. A tiny live `pufferlib.pufferl train breakout` export smoke could not be completed inside the restricted sandbox because the GPU path asserted CUDA unavailable, while `--train.gpus 0` is not supported by the current CLI path. The earlier full PufferLib training run, train-export binding hook smoke, and checkpoint exporter remain the verified working data path for this branch.

The next technical target is making this less Breakout-specific: either a learned feature/token model that discovers the intercept structure, or an explicit policy-class registry where this kind of compact fitted controller is a first-class Tokamech policy artifact. Observation-only MLP behavior cloning is not enough, and nearest-window still loses track on some seeds.
