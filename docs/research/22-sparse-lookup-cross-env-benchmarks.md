# Sparse Lookup Cross-Env Benchmarks

Date: 2026-07-06

## Purpose

Sparse lookup should be a reusable core layer, not a Snake-only workaround.

This benchmark pass wires the same policy type through all current projects:

```text
Centerline
Breakout
Snake
```

## Current benchmark result

Command:

```sh
make benchmark
```

Output:

```text
centerline_sparse_lookup steps=8 score=1 metric=1
breakout_sparse_lookup steps=220 score=2 metric=3
snake_sparse_lookup steps=96 score=11 metric=11
snake_sparse_lookup_multistart steps=288 score=34 metric=34
```

## Interpretation

Centerline:

```text
Sparse lookup learns exact raw offsets -> actions.
metric = 1 means it reaches zero offset.
```

Breakout:

```text
Sparse lookup learns compact policy tokens -> actions.
metric = paddle hits.
```

Snake:

```text
Sparse lookup learns full-state hashed tokens -> planner actions.
metric = food eaten over 96 steps.
multistart metric = aggregate food eaten over three deterministic starts.
```

## What this proves

This proves:

```text
one policy layer can run across all three current envs
TkmTrajectory can carry raw signed observations and larger token ids
small dense vocab lookup and sparse exact-token lookup are distinct useful options
benchmark tests catch whether a layer only works in one project
```

## What this does not prove

This does not prove:

```text
generalization to unseen tokens
gradient training
sequence modeling
neural policy quality
```

Sparse lookup is a strong phase-0 imitation baseline. It is useful because it makes token/data/runtime wiring explicit before replacing the policy with MLPs, transformers, Mamba, or VQ-style discrete latents.
