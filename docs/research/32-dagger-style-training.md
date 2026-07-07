# DAgger-Style Training

Date: 2026-07-06

Primary inspiration: Ross, Gordon, and Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (`https://arxiv.org/abs/1011.0686`).

## Why this matters

Plain behavior cloning trains on planner states. At evaluation time, a learned policy visits its own states. Those states can drift away from the planner rollout distribution.

DAgger addresses that mismatch by collecting states induced by the learner and asking the expert for labels on those states.

## Current benchmark implementation

This repo now has a DAgger-style Snake benchmark:

```text
initial planner behavior cloning
-> roll out the learned policy with masked argmax
-> label those induced states with snake_planner_action
-> keep training the MLP with cross entropy
-> evaluate with masked argmax and no planner fallback
```

This is not yet a reusable core trainer. It is a benchmark-level proof that the training method is useful enough to keep.

## Result

```text
snake_mlp_ce_masked_holdout steps=480 score=30 metric=30
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
snake_mlp_dagger64_masked_holdout steps=480 score=28 metric=28
snake_mlp_action_features_holdout steps=480 score=51 metric=51
```

The plain DAgger-style benchmark improves the no-planner masked MLP. The action-aware feature variant is now the best current no-planner learned Snake policy.

Planner-fallback safety is still stronger:

```text
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
```

That distinction matters. `snake_mlp_ce_safe_holdout` lets the planner choose a replacement action when the policy suggests an immediately terminal move. `snake_mlp_dagger_masked_holdout` only masks invalid actions and chooses the best remaining model logit.

## Failed first setting

The first DAgger attempt used a larger online learning rate and accidentally included a held-out first-food start in the DAgger training list. It scored:

```text
snake_mlp_dagger_masked_holdout steps=480 score=11 metric=11
```

The successful setting removed held-out leakage and reduced the online DAgger learning rate.

## Capacity experiment

A 64-hidden-unit MLP was tested with the same DAgger-style training loop:

```text
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
snake_mlp_dagger64_masked_holdout steps=480 score=28 metric=28
```

This is a negative result. The wider model completed the full held-out horizon but ate less food. For the current feature set and online update schedule, more hidden units alone is not an improvement.

The current no-planner default should remain the 32-hidden-unit DAgger-style MLP.

## Action-aware feature follow-up

The next useful DAgger improvement was not more capacity. It was a better representation.

`snake_mlp_action_features_holdout` keeps the normal Snake state vector and appends four values per possible action:

```text
valid_now
food_distance_delta
next_row_delta_to_food
next_col_delta_to_food
```

This gives the MLP direct per-action evidence before the final action head. With the same 32-hidden-unit size and DAgger-style online labeling, it improved the no-planner held-out benchmark from `40` to `51`.

## Architectural role

DAgger-style training belongs in the training layer:

```text
policy rollout -> expert label -> aggregate/update training data -> train model
```

It should eventually become a reusable core training option instead of living only in `apps/benchmark.c`.
