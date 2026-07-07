# Linear Policy

Date: 2026-07-06

## What changed

The repo now has a small trainable CPU policy model:

```text
core/model/linear_policy.h
core/model/linear_policy.c
tests/test_linear_policy.c
```

It is a multiclass linear classifier:

```text
input features -> linear logits -> argmax action
```

Training currently uses a perceptron-style update:

```text
if predicted action != target action:
    move target weights toward the input
    move predicted weights away from the input
```

This is not backprop through an MLP yet, but it is the first generic layer that changes weights from data.

## Current benchmark result

Command:

```sh
make benchmark
```

Output lines:

```text
centerline_linear_policy steps=8 score=1 metric=1
breakout_linear_policy steps=220 score=2 metric=3
```

## Why only Centerline and Breakout for now

Centerline and Breakout both have compact token-like observations that can be represented as one-hot features. A linear classifier can learn those mappings cleanly and gives a clear first training test.

Snake needs longer-horizon state and safety handling. Existing Snake results are still stronger with:

```text
sparse_lookup
nearest_policy
safety_filter
planner_oracle
```

Adding `snake_linear_policy` before it has a meaningful feature/training setup would be misleading. The right next step is either a trained MLP or a linear policy with carefully designed Snake features plus safety-intervention metrics.

## What this proves

This proves:

```text
core can train model weights on CPU
trained weights can drive env rollouts
linear_policy can be selected as a registered model kind
the benchmark can compare lookup, nearest, ngram, fixed MLP, and trained linear policies
```

## What this does not prove

This does not prove:

```text
MLP backprop works
Snake neural policy generalizes
CUDA training exists
next-token transformer training exists
```

This is the smallest useful training layer before implementing MLP backprop or sequence-model training.
