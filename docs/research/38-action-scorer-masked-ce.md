# Action Scorer Masked Cross-Entropy

Date: 2026-07-06

## What changed

The shared action scorer now has a second trainer:

```text
tkm_action_scorer_train_masked_cross_entropy
```

It keeps the same model shape as the margin scorer:

```text
score = f(state_features, action_features)
```

The difference is the loss. Instead of comparing the expert action against each other valid action one pair at a time, this trainer scores all candidate actions, applies softmax over only valid actions, and trains the expert action with cross entropy.

## Why this matters

This is a concrete test of the repo architecture goal:

```text
same project features
same action mask
same model API
different training loss
same benchmark harness
```

That is the interchangeability target. A future paper-specific loss should be able to slot in the same way.

## Result

```text
snake_action_scorer_holdout steps=480 score=58 metric=58
snake_action_scorer_ce_holdout steps=480 score=58 metric=58
```

Masked cross entropy ties the pairwise margin trainer on the current held-out Snake benchmark.

## Interpretation

This is a positive result, even though it does not beat margin.

The important finding is that changing the training objective did not require changing Snake project code, action-space features, action masking, or evaluation. The new loss is a drop-in option on the same scorer layer.

## Current limit

The trainer is available in C. A training config parser recognizes `margin` and `masked_cross_entropy`, and a core action-scorer trainer dispatch now routes those loss choices through one API. Full project-runtime wiring from a complete INI file is still separate follow-up work. See `39-train-config-loss-selection.md` and `40-action-scorer-trainer-dispatch.md`.
