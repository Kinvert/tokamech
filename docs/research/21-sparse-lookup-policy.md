# Sparse Lookup Policy

Date: 2026-07-06

## What changed

The repo now has a generic sparse lookup policy:

```text
core/train/sparse_lookup_policy.h
core/train/sparse_lookup_policy.c
tests/test_sparse_lookup_policy.c
```

It maps exact `int32_t` observation/token keys to actions by majority vote over a trajectory.

## Why this exists

The original dense lookup policy is good for small vocabularies:

```text
token id -> action count table
```

That works for Centerline, Breakout, and small compact Snake tokens, but it does not work well when a project needs large or hashed state tokens.

Snake exposed that problem:

```text
compact token: direction + food direction + local danger
result: one food, then weak play
```

Adding head and food cells helped, but body geometry still caused token collisions. The same head/food/direction can require different actions depending on the snake body.

## Current sparse policy behavior

Training:

```text
for each trajectory item:
    find exact token
    count the action for that token
choose the most common action per token
```

Prediction:

```text
if token was seen during training:
    return learned action
else:
    return configured default action
```

This is intentionally simple:

```text
no hashing table
no allocation
no neural network
no hidden state
```

The current max unique token count is `512`, matching the phase-0 trajectory size.

## Snake full-state token

Snake now has a full-state token for sparse imitation:

```text
snake_tokenize_full_observation
```

It hashes:

```text
head row/col
food row/col
direction
length
body cells
```

The hash is masked to a non-negative `int32_t` so it can live in `TkmTrajectory`.

## Benchmark result

Command:

```sh
make benchmark
```

Output line:

```text
centerline_sparse_lookup steps=8 score=1 metric=1
breakout_sparse_lookup steps=220 score=2 metric=3
snake_sparse_lookup steps=96 score=11 metric=11
snake_sparse_lookup_multistart steps=288 score=34 metric=34
```

This matches:

```text
snake_planner steps=96 score=11 metric=11
```

## What this proves

This proves:

```text
planner rollouts can be stored as token/action data
the same sparse policy works across Centerline, Breakout, and Snake
large exact tokens can be learned without dense vocab allocation
Snake can play the full benchmark through a learned lookup table
the benchmark can distinguish weak compact tokens from richer full-state tokens
```

## What this does not prove

This does not prove:

```text
generalization to unseen Snake starts
neural policy training
transformer-style next-token prediction
sim2real transfer
```

It is still a useful step because it creates a strong learned baseline before replacing the policy layer with MLP, transformer, Mamba, or another model.
