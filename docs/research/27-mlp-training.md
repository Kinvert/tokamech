# MLP Training

Date: 2026-07-06

## What changed

`mlp_window` now supports one-sample CPU training:

```text
tkm_mlp_window_train_mse
```

The training path performs a deterministic SGD update through:

```text
linear -> ReLU -> linear logits
```

The target is a one-hot action vector and the loss is mean-squared error over logits.

## Current benchmark result

Command:

```sh
make benchmark
```

Output line:

```text
centerline_mlp_trained steps=8 score=1 metric=1
```

## Why Centerline first

Centerline is small enough that the MLP training path can be audited by hand:

```text
offset < 0 -> RIGHT
offset = 0 -> STAY
offset > 0 -> LEFT
```

The benchmark uses deterministic initial hidden features and trains output behavior from oracle data.

## What this proves

This proves:

```text
core can update MLP weights from data
trained MLP weights can drive an environment rollout
the benchmark can distinguish fixed-weight MLP from trained MLP
```

## What this does not prove

This does not prove:

```text
mini-batch training
cross-entropy backprop
optimizer support
Snake trained-neural policy quality
CUDA kernels
sequence-model training
```

The next useful step is to train an MLP from richer Snake features and report both policy score and safety-filter interventions. If that fails, the benchmark should keep the failure visible rather than hiding it behind the planner.
