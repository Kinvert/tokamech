# Nearest Policy

Date: 2026-07-06

## What changed

The repo now has a generic continuous nearest-neighbor policy:

```text
core/model/nearest_policy.h
core/model/nearest_policy.c
tests/test_nearest_policy.c
```

The layer stores fixed-size float feature vectors with action labels. At inference, it picks the action from the closest stored vector by squared Euclidean distance.

## Why this layer exists

Sparse lookup is useful when exact token identity matters. It does not generalize to nearby continuous states.

Nearest neighbor is the smallest useful continuous policy baseline:

```text
feature vector -> closest stored example -> action
```

It is not a neural network and it does not train weights, but it exercises the same broad data shape that future MLP, transformer, or Mamba layers need:

```text
project state -> continuous features -> policy model -> action
```

## Snake feature vector

Snake now exposes:

```text
snake_write_features
```

The feature vector contains:

```text
normalized head row/col
normalized food row/col
normalized food delta
direction one-hot
normalized length
terminal flag
signed grid occupancy
```

Grid occupancy uses:

```text
body = 1
food = -1
empty/wall = 0
```

This is intentionally simple and auditable.

## Current benchmark result

Command:

```sh
make benchmark
```

Output lines:

```text
centerline_nearest steps=8 score=1 metric=1
breakout_nearest steps=220 score=2 metric=3
snake_nearest_multistart steps=288 score=34 metric=34
snake_nearest_safe_holdout steps=480 score=47 metric=47
```

## What this proves

This proves:

```text
continuous feature policies work in the core stack
the same nearest-neighbor model can run on Centerline, Breakout, and Snake
Snake can be driven by continuous features instead of exact full-state token lookup
nearest policy can match the multistart sparse lookup benchmark on trained starts
nearest policy plus an immediate safety filter can survive held-out first-food starts
```

## What this does not prove

This does not prove:

```text
pure held-out random Snake generalization without a safety filter
gradient training
CUDA acceleration
next-token sequence modeling
```

Nearest alone performs poorly on several held-out starts. The safety-filtered benchmark is useful because it separates action suggestion from immediate actuation safety. The next useful step is a learned model that reduces how often the safety filter has to intervene.
