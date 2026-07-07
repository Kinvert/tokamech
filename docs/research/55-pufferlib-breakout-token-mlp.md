# PufferLib Breakout Token MLP

Date: 2026-07-07.

## Goal

Build a real Tokamech learned policy for literal PufferLib Breakout without the hand-coded paddle/ball intercept shortcut. The data source is the exported PufferLib Breakout JSONL:

```text
/tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
```

That file has `100000` rows, `obs_dim=118`, and was exported from a trained PufferLib checkpoint with frameskip 4.

## Model

The working learned policy is `token_mlp_all` in `projects/breakout/pufferlib_policy.c`.

It is generic and data-driven:

```text
JSONL rows
-> validate/load 118-float PufferLib observations and action tokens
-> rank observation dimensions by action-label separation
-> selected continuous observation tokens normalized from dataset min/max
-> current token frame + 8 previous observation-token frames
-> one-hidden-layer MLP, hidden=64
-> predict next action token in {0,1,2}
-> run action directly in PufferLib Breakout c_step
```

It does not compute paddle intercepts, wall reflections, or ball landing positions. The old `intercept_rule` and `intercept_trained` paths remain baselines only.

Default benchmark/render token settings:

```text
TKM_PUFFERLIB_BREAKOUT_TOKEN_BINS=16
TKM_PUFFERLIB_BREAKOUT_TOKEN_DIMS=16
TKM_PUFFERLIB_BREAKOUT_TOKEN_EPOCHS=80
TKM_PUFFERLIB_BREAKOUT_FRAMESKIP=4
```

## Results

Command shape:

```sh
TKM_PUFFERLIB_BREAKOUT_FRAMESKIP=4 \
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=3 \
TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=50000 \
TKM_PUFFERLIB_BREAKOUT_EVAL_SEED=<seed> \
./build/pufferlib_breakout_benchmark \
  /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
  /tmp/tkm_token_cont_seed<seed>_e80
```

No-render PufferLib C eval:

```text
seed  token_mlp_all mean_return  max_return  mean_length  actions
0     717.0                      864.0       3673.0       [2745,4091,4183]
1     864.0                      864.0       5365.3       [3433,6746,5917]
2     847.7                      864.0       4670.3       [3249,5570,5192]
```

Mean over seeds 0-2: `809.6`.

Baseline context on the same eval shape:

```text
intercept_trained remains 864.0 on these seeds, but is hand-coded Breakout geometry.
nearest/sequence retrieval is generic but remains a baseline/replay method.
raw mlp_all with the benchmark default remains poor, single-digit returns.
token_mlp_top1 from only the top episode fails, generally low double-digit return.
```

Held-out action accuracy for `token_mlp_all` at the 80-epoch setting was `0.717`. The important result is the PufferLib rollout return, not the accuracy alone.

## Render

`apps/pufferlib_breakout_policy_render.c` now defaults to `token_mlp`, not `intercept_trained`.

Default render mode:

```sh
TKM_PUFFERLIB_BREAKOUT_JSONL=/tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
DISPLAY=:0 ./build/pufferlib_breakout_policy_render
```

Baseline modes remain explicit:

```text
TKM_PUFFERLIB_RENDER_MODE=intercept_trained
TKM_PUFFERLIB_RENDER_MODE=intercept_rule
TKM_PUFFERLIB_RENDER_MODE=sequence
TKM_PUFFERLIB_RENDER_MODE=nearest
TKM_PUFFERLIB_RENDER_MODE=mlp
```

Display commands are not part of unattended gates.

## Next Checks

Run a display check only when explicitly requested. For headless verification, keep using `pufferlib_breakout_benchmark` with multiple seeds and report score/return/length/action histogram.
