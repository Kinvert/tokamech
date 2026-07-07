# PufferLib Breakout Layered Token Stack

Date: 2026-07-07.

## Goal

Turn the working PufferLib Breakout `token_mlp` path into a small configurable Tokamech stack. The data source is still literal PufferLib Breakout JSONL, and the policy still predicts the next action token from observations and recent action/observation history. The change is that the stack is selected by INI instead of being a project-local one-off.

The known-good MLP remains selectable as `sequence_model.kind = mlp_window`. It is the default. The first alternate backend is `sequence_model.kind = linear_policy`, which uses the same dataset loader, sequence layout, tokenizer, and action head.

The old `nearest`, `sequence_cursor`, raw `mlp`, `intercept_rule`, and `intercept_trained` paths remain explicit baselines. The intercept baselines are not the main policy because they use Breakout-specific geometry.

## Layer Contract

Current config layers:

```text
[breakout.pufferlib]
dataset = /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
frameskip = 4

[sequence_layout]
kind = obs_action_interleaved
history = 8
predict = action

[tokenizer.observation]
kind = selected_continuous
dims = 16
selection = action_separation
bins = 16

[sequence_model]
kind = mlp_window
hidden = 64
epochs = 80
learning_rate = 0.01

[action_head]
kind = categorical
actions = 3
```

Implemented config files:

```text
projects/breakout/pufferlib_token_mlp.ini
projects/breakout/pufferlib_token_linear.ini
```

Both benchmark and render read the token stack from:

```text
TKM_PUFFERLIB_BREAKOUT_CONFIG
```

If `TKM_PUFFERLIB_BREAKOUT_JSONL` is not set, both use the dataset path from `[breakout.pufferlib]`.

## Benchmark

Build:

```sh
make -B build/pufferlib_breakout_benchmark
```

Default MLP stack:

```sh
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_mlp.ini \
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=3 \
TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=50000 \
TKM_PUFFERLIB_BREAKOUT_EVAL_SEED=0 \
./build/pufferlib_breakout_benchmark \
  /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
  /tmp/tkm_v002_mlp_seed0_e3
```

Observed no-render PufferLib C eval:

```text
token_layers layout=obs_action_interleaved tokenizer=selected_continuous model=mlp_window head=categorical
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.717 bins=16 dims=16 history=8 epochs=80 learning_rate=0.01000 input_dim=144 hidden=64 pair_features=1
method=token_mlp_all rows=100000 episodes=3 mean_return=719.000 max_return=864.000 mean_length=3855.333 steps=11566 actions=[2741,4232,4593]
```

This matches the v001 behavior class. The previous seed-0 record was `717.0` mean return over three episodes; small differences come from the current benchmark/default training path, but the policy is still the same token MLP stack and still strong in PufferLib rollout.

Alternate linear backend:

```sh
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_linear.ini \
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=1 \
TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=8000 \
TKM_PUFFERLIB_BREAKOUT_EVAL_SEED=0 \
./build/pufferlib_breakout_benchmark \
  /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
  /tmp/tkm_v002_linear_smoke
```

Observed result:

```text
token_layers layout=obs_action_interleaved tokenizer=selected_continuous model=linear_policy head=categorical
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.684 bins=16 dims=16 history=8 epochs=8 learning_rate=0.20000 input_dim=0 hidden=64 pair_features=1
method=token_linear_policy_all rows=100000 episodes=1 mean_return=5.000 max_return=5.000 mean_length=291.000 steps=291 actions=[102,106,83]
```

The linear backend is intentionally a swap proof, not a performance claim. It proves the PufferLib adapter and JSONL loader do not need to change when the sequence model changes.

## Render

Build headlessly:

```sh
make -B build/pufferlib_breakout_policy_render
```

Display run, only when explicitly requested:

```sh
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_mlp.ini \
DISPLAY=:0 ./build/pufferlib_breakout_policy_render
```

Selectable render modes:

```text
TKM_PUFFERLIB_RENDER_MODE=token
TKM_PUFFERLIB_RENDER_MODE=token_mlp
TKM_PUFFERLIB_RENDER_MODE=nearest
TKM_PUFFERLIB_RENDER_MODE=sequence_cursor
TKM_PUFFERLIB_RENDER_MODE=intercept_rule
TKM_PUFFERLIB_RENDER_MODE=intercept_trained
TKM_PUFFERLIB_RENDER_MODE=mlp
```

The token backend is selected by `TKM_PUFFERLIB_BREAKOUT_CONFIG`, not by render mode. Render prints the token config path, dataset path, frameskip, layer names, training report, and final action histogram. Display commands stay out of unattended gates.

## Current Limits

This is still not a transformer-style token model. The useful v002 step is the layer boundary: the dataset loader, sequence layout, tokenizer, model, and head are now named and selectable. Future paper-inspired backends should be added behind `sequence_model.kind` or adjacent layer keys without changing the PufferLib adapter.
