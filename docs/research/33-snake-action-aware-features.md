# Snake Action-Aware Features

Date: 2026-07-06

## Why this was added

The prior best no-planner Snake policy used normal continuous state features, DAgger-style online planner labels, and masked argmax:

```text
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
```

A wider 64-hidden-unit MLP made that worse:

```text
snake_mlp_dagger64_masked_holdout steps=480 score=28 metric=28
```

That suggested the next improvement should be representation, not raw capacity.

## Feature layout

`snake_write_action_features` writes the existing Snake feature vector first:

```text
state features: head, food, direction, length, terminal flag, board occupancy
```

It then appends four values for each candidate action:

```text
valid_now
food_distance_delta
next_row_delta_to_food
next_col_delta_to_food
```

The layout stays explicit C code. It is not a black box tokenizer. A reader can see exactly how each feature is written.

## Result

```text
snake_mlp_action_features_holdout steps=480 score=51 metric=51
```

This is the best current no-planner-fallback learned Snake policy.

## Interpretation

The MLP now receives direct per-action evidence before producing action logits. That is a useful inductive bias for small models:

```text
candidate action -> immediate safety and food direction evidence -> action logit
```

The policy still uses masked argmax at evaluation time, so immediately terminal moves are blocked before action selection. It does not use the planner as an evaluation fallback.

## Architectural lesson

This is a good example of why the repo needs interchangeable representation layers.

The model, loss, DAgger-style training loop, and action head stayed basically the same. The useful change was the project-level feature writer:

```text
snake_write_features -> snake_write_action_features
```

Future projects should be able to make the same kind of swap through a config-selected representation path once this benchmark-level code graduates into reusable core interfaces.
