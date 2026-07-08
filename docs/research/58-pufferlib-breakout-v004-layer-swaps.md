# PufferLib Breakout v004 Layer Swaps

Date: 2026-07-07.

## Goal

Add Tokamech-only drop-in methods behind the existing PufferLib Breakout JSONL
pipeline. This keeps Pathfinder/PufferLib unchanged and keeps the real
PufferLib Breakout loop as the eval authority.

Existing methods remain selectable:

```text
mlp_window
linear_policy
token_ngram
```

New v004 methods:

```text
token_backoff_ngram
action_mlp_from_token_context
token_mlp_window
```

Additional v004 layer options:

```text
tokenizer.observation.selection = first_dims
tokenizer.observation.selection = manual
action_head.kind = typed_next_token_unmasked
```

The benchmark and render adapters print the tokenizer-selection layer as
`selection=...` in the `token_layers` line.

## New Methods

`token_backoff_ngram` keeps the v003 typed next-token table but changes missing
context behavior. Action prediction now tries exact context, then shorter suffix
contexts, then the learned action prior. The report prints `exact`, `backoff`,
and `prior` counts.

`action_mlp_from_token_context` is a bridge method. It uses the
`autoregressive_obs_action` stream and `selected_quantized` observation tokens,
but predicts the action directly with a categorical head. It is not a full
next-token model.

`token_mlp_window` is the learned typed next-token backend. It consumes the same
typed stream context and predicts either OBS-token classes or ACTION-token
classes with masked cross-entropy. Runtime inserts real PufferLib observation
tokens, then executes the predicted ACTION token.

`first_dims` and `manual` are tokenizer-selection alternatives. `first_dims`
uses dimensions `0..dims-1`. `manual` takes a comma-separated
`tokenizer.observation.indices` list and uses those observation dimensions
exactly.

`typed_next_token_unmasked` is a prediction-head/loss alternative for
`token_mlp_window`. It trains over the full typed vocabulary instead of using
OBS-only and ACTION-only masks at each target.

## Configs

```text
projects/breakout/pufferlib_token_backoff_ngram.ini
projects/breakout/pufferlib_token_action_mlp_context.ini
projects/breakout/pufferlib_token_mlp_window.ini
projects/breakout/pufferlib_token_action_mlp_first_dims.ini
projects/breakout/pufferlib_token_action_mlp_manual.ini
projects/breakout/pufferlib_token_mlp_window_unmasked.ini
```

## Permutation Checks

The unit test suite now includes a synthetic layer matrix for the current
PufferLib Breakout token stack. It verifies that supported combinations of
layout, tokenizer, tokenizer selection, sequence model, and head can parse,
train, report layer names, and produce an action. It also verifies that invalid
cross-layer combinations are rejected at config load time.

For real PufferLib no-render smoke coverage, run:

```sh
make pufferlib-breakout-matrix-smoke \
  TKM_PUFFERLIB_BREAKOUT_JSONL=/tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
```

That target loops all checked-in `projects/breakout/pufferlib_token_*.ini`
configs through `build/pufferlib_breakout_benchmark`. Use
`TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES`, `TKM_PUFFERLIB_BREAKOUT_MAX_STEPS`,
`TKM_PUFFERLIB_BREAKOUT_EVAL_SEED`, and `TKM_PUFFERLIB_BREAKOUT_MATRIX_OUT` to
control the run.

## Smoke Results

Dataset:

```text
/tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
```

All commands used one PufferLib eval episode, seed 0, and `max_steps=8000`.

`token_backoff_ngram`:

```text
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.343 obs_token_accuracy=0.407 action_token_accuracy=0.343 exact=728 backoff=19272 prior=0
method=token_backoff_ngram_all rows=100000 episodes=1 mean_return=0.000 mean_length=116.000 actions=[84,24,8]
method=token_backoff_ngram_top1 rows=4560 episodes=1 mean_return=7.000 mean_length=363.000 actions=[0,0,363]
```

`action_mlp_from_token_context`:

```text
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.670 obs_token_accuracy=0.000 action_token_accuracy=0.670 input_dim=16 hidden=64
method=token_action_mlp_context_all rows=100000 episodes=1 mean_return=9.000 mean_length=414.000 actions=[254,56,104]
method=token_action_mlp_context_top1 rows=4560 episodes=1 mean_return=7.000 mean_length=363.000 actions=[0,305,58]
```

`token_mlp_window`:

```text
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.629 obs_token_accuracy=0.591 action_token_accuracy=0.629 input_dim=16 hidden=64
method=token_mlp_window_all rows=100000 episodes=1 mean_return=9.000 mean_length=331.000 actions=[208,64,59]
method=token_mlp_window_top1 rows=4560 episodes=1 mean_return=15.000 mean_length=662.000 actions=[0,309,353]
```

`action_mlp_from_token_context` with `selection = first_dims`:

```text
token_selected_dims=0,1,2,3,4,5,6,7
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.644 action_token_accuracy=0.644
method=token_action_mlp_context_all rows=100000 episodes=1 mean_return=6.000 mean_length=250.000 actions=[137,43,70]
method=token_action_mlp_context_top1 rows=4560 episodes=1 mean_return=14.000 mean_length=609.000 actions=[58,197,354]
```

`action_mlp_from_token_context` with `selection = manual`:

```text
token_selected_dims=3,4,2,15,13,17,14,12
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.670 action_token_accuracy=0.670
method=token_action_mlp_context_all rows=100000 episodes=1 mean_return=9.000 mean_length=414.000 actions=[254,56,104]
method=token_action_mlp_context_top1 rows=4560 episodes=1 mean_return=7.000 mean_length=363.000 actions=[0,349,14]
```

`token_mlp_window` with `typed_next_token_unmasked`:

```text
token_layers layout=autoregressive_obs_action tokenizer=selected_quantized selection=action_separation model=token_mlp_window head=typed_next_token_unmasked
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.621 obs_token_accuracy=0.591 action_token_accuracy=0.621
method=token_mlp_window_all rows=100000 episodes=1 mean_return=9.000 mean_length=331.000 actions=[209,64,58]
method=token_mlp_window_top1 rows=4560 episodes=1 mean_return=14.000 mean_length=609.000 actions=[0,376,233]
```

## Readout

The important result is modularity, not strong Breakout play. Three different
sequence-model choices now run through the same JSONL loader, tokenizer,
training report, and no-render PufferLib eval path. The tokenizer-selection
layer and typed-head/loss layer now also have INI-selectable alternatives.

The learned token MLP is the closest v004 method to the main next-token framing:
it trains both observation-token and action-token predictions under the typed
stream contract. It still uses a tiny MLP and an 8-token context, so it should
not be read as a transformer-scale result.

The backoff n-gram proves that fallback behavior is now visible instead of
hidden. On the 100k-row smoke, nearly all heldout actions used suffix backoff
rather than the prior.

Current failure modes:

```text
short context
small model capacity
no return conditioning
no planner/decoder over predicted futures
behavior cloning only
weak Breakout return despite reasonable heldout action accuracy
```
