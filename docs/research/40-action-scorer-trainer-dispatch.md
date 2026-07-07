# Action Scorer Trainer Dispatch

Date: 2026-07-06

## What changed

The repo now has a small core trainer dispatch for the shared action scorer:

```text
core/train/action_scorer_trainer.h
core/train/action_scorer_trainer.c
```

The public entry point is:

```text
tkm_action_scorer_train_configured
```

It receives:

```text
TkmActionScorer
TkmTrainLossKind
state features
candidate action features
valid-action mask
target action
margin
learning rate
```

## Supported dispatch choices

```text
TKM_TRAIN_LOSS_MARGIN -> tkm_action_scorer_train_margin
TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY -> tkm_action_scorer_train_masked_cross_entropy
```

Other loss kinds currently return `TKM_ERR` for this model.

That is intentional. `cross_entropy` and `mse` are valid training losses in the repo, but they are not valid objectives for this scalar action-conditioned scorer unless a specific adapter is implemented.

## Why this matters

This turns the previous parser-only seam into a real layer boundary:

```text
INI parses train.loss
core trainer dispatch receives TkmTrainLossKind
project writes explicit state/action features
core trainer applies the selected objective
```

The project still owns readable feature-writing code. Core owns the reusable model/loss dispatch. That is the intended split.

## Snake benchmark wiring

The Snake action-scorer benchmark now routes both scorer training paths through the dispatch:

```text
snake_action_scorer_holdout -> TKM_TRAIN_LOSS_MARGIN
snake_action_scorer_ce_holdout -> TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY
snake_action_scorer_config_margin_holdout -> parsed [train] loss = margin
snake_action_scorer_config_ce_holdout -> parsed [train] loss = masked_cross_entropy
```

The benchmark names and behavior stay the same, but the route is now closer to the target configurable architecture.

## Result

```text
snake_action_scorer_holdout steps=480 score=58 metric=58
snake_action_scorer_ce_holdout steps=480 score=58 metric=58
snake_action_scorer_config_margin_holdout steps=480 score=58 metric=58
snake_action_scorer_config_ce_holdout steps=480 score=58 metric=58
```

Centerline and Breakout benchmark rows were unchanged by this dispatch layer.

## Tests

Coverage was added in:

```text
tests/test_action_scorer_trainer.c
```

The tests verify:

```text
masked_cross_entropy dispatch learns the target action
margin dispatch learns the target action
unsupported loss kinds fail
invalid masks and targets fail
```

## Current limit

This is still a per-state trainer call. It does not own rollout collection, batching, DAgger scheduling, optimizer state, or checkpointing.

The config-driven Snake benchmark is also still a benchmark seam, not a full project runner. It proves `train.loss` can choose the action-scorer objective from INI text, but the repo still needs a project-level training runner that owns data collection and checkpoints.
