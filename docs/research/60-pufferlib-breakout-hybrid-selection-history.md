# PufferLib Breakout Hybrid Selection and History Sweep

Date: 2026-07-08.

## Goal

Try another layer idea after `pufferlib_token_mlp_first_dims.ini` became the
best full-row policy. The target was to beat the benchmark by adding more
vectorized observation dimensions without losing the useful compact Breakout
state prefix.

## New Selection Layer

Added a tokenizer-selection option:

```ini
[tokenizer.observation]
selection = first_then_action_separation
prefix_dims = 16
```

This keeps dimensions `0..prefix_dims-1`, then fills the remaining selected
slots using the existing action-separation score over unselected dimensions.

Config:

```text
projects/breakout/pufferlib_token_mlp_first_then_action.ini
```

## Additional Configs

To isolate whether the result was from more dimensions or from the hybrid
scoring rule, also added:

```text
projects/breakout/pufferlib_token_mlp_first_dims24.ini
projects/breakout/pufferlib_token_mlp_first_dims24_h4.ini
projects/breakout/pufferlib_token_mlp_first_dims_h4.ini
```

## Benchmark

Command shape:

```sh
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=3 \
TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=8000 \
TKM_PUFFERLIB_BREAKOUT_CONFIG=<config> \
./build/pufferlib_breakout_benchmark \
  /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
  /tmp/tkm_priority_breakout_benchmark/<name>
```

Current full-row anchor:

```text
config=projects/breakout/pufferlib_token_mlp_first_dims.ini
selection=first_dims dims=16 history=8 input_dim=144
heldout_accuracy=0.746
method=token_mlp_first_dims_all episodes=2 mean_return=864.000 max_return=864.000 mean_length=3636.000
method=token_mlp_first_dims_top1 episodes=3 mean_return=14.000 max_return=16.000 mean_length=409.333
```

Hybrid selector:

```text
config=projects/breakout/pufferlib_token_mlp_first_then_action.ini
selection=first_then_action_separation dims=24 history=8 input_dim=216
token_selected_dims=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,16,30,32,31,18,35,34
heldout_accuracy=0.762
method=token_mlp_first_then_action_all episodes=2 mean_return=757.000 max_return=864.000 mean_length=3842.500
method=token_mlp_first_then_action_top1 episodes=3 mean_return=141.000 max_return=187.000 mean_length=1046.667
```

First 24 dims, history 8:

```text
config=projects/breakout/pufferlib_token_mlp_first_dims24.ini
selection=first_dims dims=24 history=8 input_dim=216
heldout_accuracy=0.780
method=token_mlp_first_dims_all episodes=1 mean_return=864.000 max_return=864.000 mean_length=5226.000
method=token_mlp_first_dims_top1 episodes=3 mean_return=203.000 max_return=327.000 mean_length=1269.667
```

First 24 dims, history 4:

```text
config=projects/breakout/pufferlib_token_mlp_first_dims24_h4.ini
selection=first_dims dims=24 history=4 input_dim=120
heldout_accuracy=0.791
method=token_mlp_first_dims_all episodes=1 mean_return=864.000 max_return=864.000 mean_length=5287.000
method=token_mlp_first_dims_top1 episodes=3 mean_return=311.333 max_return=414.000 mean_length=1885.667
```

First 16 dims, history 4:

```text
config=projects/breakout/pufferlib_token_mlp_first_dims_h4.ini
selection=first_dims dims=16 history=4 input_dim=80
heldout_accuracy=0.775
method=token_mlp_first_dims_all episodes=1 mean_return=864.000 max_return=864.000 mean_length=4596.000
method=token_mlp_first_dims_top1 episodes=3 mean_return=316.333 max_return=411.000 mean_length=2364.000
```

## Readout

The new hybrid selection layer did not beat the full-row first-dims anchor.
Adding action-separation tail dimensions improved heldout action accuracy but
reduced full-row score.

The highest full-row policy remains:

```text
projects/breakout/pufferlib_token_mlp_first_dims.ini
```

The best top-return-filtered policy in this sweep is:

```text
projects/breakout/pufferlib_token_mlp_first_dims_h4.ini
```

That is a meaningful result for the modular stack: the best full-row and best
top1 policies now prefer different sequence-history settings, and the new
selection layer ran end to end even though it was not the winner.
