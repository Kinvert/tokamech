# Action Scorer Benchmark

Date: 2026-07-06

## What was implemented

The repo now has a reusable core action scorer:

```text
core/model/action_scorer.h
core/model/action_scorer.c
```

It scores one candidate action at a time:

```text
score = f(state_features, action_features)
```

A masked argmax then chooses the highest-scoring valid action.

## Training

The first trainer is pairwise margin imitation:

```text
expert action score should be higher than each other valid action score by margin
```

For Snake, the expert is still `snake_planner_action`. Training uses planner rollouts plus DAgger-style policy-induced states.

## Representation split

The existing appended-action MLP path keeps:

```text
snake_write_action_features
valid_now
food_distance_delta
next_row_delta_to_food
next_col_delta_to_food
```

The new shared scorer path uses:

```text
snake_write_action_space_features
valid_now
food_distance_delta
next_row_delta_to_food
next_col_delta_to_food
reachable_space_fraction_after_action
```

This split is important. Mutating the old representation caused a regression:

```text
snake_mlp_action_features_holdout steps=480 score=38 metric=38
```

Keeping the old representation and adding a separate action-space representation restored the old MLP result and let the scorer improve:

```text
snake_mlp_action_features_holdout steps=480 score=51 metric=51
snake_action_scorer_holdout steps=480 score=58 metric=58
```

## Result

```text
snake_action_scorer_holdout steps=480 score=58 metric=58
```

This is the best current no-planner-fallback learned Snake policy.

## Lesson

The architecture should keep representations interchangeable.

Adding one feature can help one model and hurt another. That is exactly why the repo should support selectable representation/tokenizer/vectorizer paths instead of forcing every model through one global feature layout.

## Next steps

Good follow-up options:

```text
add INI selection for action_scorer
move DAgger-style training out of apps/benchmark.c into reusable core training code
```

Centerline and Breakout scorer benchmarks were added next; see `36-action-scorer-cross-project.md`.
Masked score cross-entropy was added next; see `38-action-scorer-masked-ce.md`.
