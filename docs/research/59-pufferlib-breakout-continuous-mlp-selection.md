# PufferLib Breakout Continuous MLP Selection Sweep

Date: 2026-07-08.

## Goal

Try layer changes that could beat the previous best `token_mlp_all` result while
keeping the PufferLib Breakout JSONL and no-render eval path authoritative.

## New Input-Feature Layer

Added an INI-selectable input-feature layer for the continuous `mlp_window`
policy:

```ini
[input_features]
kind = delta_action_history
```

`value_window` remains the default and preserves the existing input contract.
`delta_action_history` appends one-step normalized observation deltas and
previous-action one-hot history to the existing selected observation window.

Config:

```text
projects/breakout/pufferlib_token_mlp_delta_action.ini
```

## Selection Benchmark

The stronger result came from using the existing tokenizer-selection layer with
the continuous MLP:

```ini
[tokenizer.observation]
selection = first_dims
```

Config:

```text
projects/breakout/pufferlib_token_mlp_first_dims.ini
```

This keeps the first 16 Breakout observation fields, which include the compact
game-state features before the brick-state array.

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

Baseline from the same settings:

```text
config=projects/breakout/pufferlib_token_mlp.ini
token_layers layout=obs_action_interleaved tokenizer=selected_continuous selection=action_separation input=value_window model=mlp_window head=categorical
token_train heldout_accuracy=0.717 input_dim=144
method=token_mlp_all rows=100000 episodes=2 mean_return=657.000 max_return=864.000
```

New input-feature layer:

```text
config=projects/breakout/pufferlib_token_mlp_delta_action.ini
token_layers layout=obs_action_interleaved tokenizer=selected_continuous selection=action_separation input=delta_action_history model=mlp_window head=categorical
token_train heldout_accuracy=0.740 input_dim=184
method=token_mlp_delta_action_all rows=100000 episodes=2 mean_return=525.500 max_return=701.000
```

Selection-layer win:

```text
config=projects/breakout/pufferlib_token_mlp_first_dims.ini
token_layers layout=obs_action_interleaved tokenizer=selected_continuous selection=first_dims input=value_window model=mlp_window head=categorical
token_train heldout_accuracy=0.746 input_dim=144
token_selected_dims=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
method=token_mlp_first_dims_all rows=100000 episodes=2 mean_return=864.000 max_return=864.000
```

## Readout

`delta_action_history` is useful as a modular layer proof, but it did not beat
the baseline score despite better heldout action accuracy. It likely overfits
short-horizon behavior more than it improves game control.

`first_dims` is the current best Tokamech policy combination for PufferLib
Breakout under this benchmark. It beats the previous action-separation MLP
anchor by keeping the compact game-state observation fields instead of mostly
selecting high-variance brick dimensions.
