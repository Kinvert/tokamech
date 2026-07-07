# Sequence Layout Layer

Date: 2026-07-06

## Current implementation status

The first sequence layout module is implemented in core:

```text
core/sequence/layout.h
core/sequence/layout.c
tests/test_sequence_layout.c
```

Implemented builders:

```text
tkm_sequence_build_obs_action_interleaved
tkm_sequence_build_joint_transitions
```

Implemented item types:

```text
PAD
BOS
EOS
OBS
ACTION
REWARD
TERMINAL
TRANSITION
```

## What sequence layout means

Sequence layout is not tokenization.

Tokenizer:

```text
raw project data -> token IDs
```

Sequence layout:

```text
already-tokenized fields -> ordered training/runtime stream
```

Example:

```text
obs token = 10
action token = 1
reward token = 5
terminal token = 0
```

Obs/action interleaved layout:

```text
BOS OBS(10) ACTION(1) REWARD(5) TERMINAL(0) EOS
```

## Implemented layouts

Obs/action interleaved:

```ini
[sequence_layout]
kind = obs_action_interleaved
```

Stream shape:

```text
BOS
OBS(t0) ACTION(t0) REWARD(t0) TERMINAL(t0)
OBS(t1) ACTION(t1) REWARD(t1) TERMINAL(t1)
EOS
```

This is the clearest default for learning and debugging because every item keeps a type.

Joint transition:

```ini
[sequence_layout]
kind = joint_transition
```

Stream shape:

```text
BOS
TRANSITION(t0)
TRANSITION(t1)
EOS
```

This supports methods where one packed token represents a whole transition or state-action bundle.

## Why this layer matters

Different papers make different choices about what becomes the next token.

Some predict:

```text
next action token
```

Some predict:

```text
next observation token
```

Some pack:

```text
observation + action + reward + terminal -> one transition token
```

Tokamech should allow those layout choices without rewriting project envs.

## Current tests

`tests/test_sequence_layout.c` covers:

```text
obs/action/reward/terminal interleaving
joint transition stream construction
required item counts
bad inputs and short output buffers
```

## Planned layouts

Useful future layouts:

```text
action_only_prediction
obs_then_action
decision_transformer_return_obs_action
masked_token_layout
multi_modal_rt_style
sliding_window_views
episode_packing_with_reset
```

## Architecture rule

Core sequence layout should only know typed token IDs.

Good:

```text
OBS(10) ACTION(2) REWARD(1) TERMINAL(0)
```

Bad:

```text
QTI_LEFT_ON_LINE ACTION_TURN_RIGHT
SNAKE_FOOD_NEAR_HEAD
BREAKOUT_PADDLE_HIT
```

Project-specific meaning belongs in project observation/tokenization code and docs. Core layout only controls item order and type tags.
