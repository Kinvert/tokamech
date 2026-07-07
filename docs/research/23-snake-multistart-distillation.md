# Snake Multistart Distillation

Date: 2026-07-06

## Why this was added

The first strong Snake learned baseline used one deterministic start:

```text
snake_sparse_lookup steps=96 score=11 metric=11
```

That proved the sparse lookup layer could replay a competent planner rollout, but it was still too narrow. A Snake policy that only works from one fixed first food position is not a strong enough benchmark for "plays well."

## New benchmark

The multistart benchmark trains one sparse lookup policy from planner rollouts that begin with three different first-food positions:

```text
{6, 8}
{3, 5}
{8, 5}
```

Each scenario runs for 96 steps. The benchmark reports aggregate steps and aggregate food.

Command:

```sh
make benchmark
```

Output line:

```text
snake_sparse_lookup_multistart steps=288 score=34 metric=34
```

## What this proves

This proves:

```text
full-state sparse tokens can store multiple planner-distilled Snake rollouts
one policy can replay several deterministic Snake starts without terminal
the Snake benchmark is no longer only a single fixed-food memorization check
```

## What this does not prove

This does not prove:

```text
generalization to arbitrary random starts
robustness to sensor noise
neural policy training
next-token sequence modeling
```

The next Snake-quality step should be either:

```text
evaluate held-out first-food positions
add a continuous feature policy that can generalize across positions
train a small MLP classifier from planner data
replace exact lookup with a sequence model over observation/action history
```
