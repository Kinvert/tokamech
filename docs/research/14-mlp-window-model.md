# MLP Window Model

Date: 2026-07-06

## Current implementation status

The first generic model module is implemented in core:

```text
core/model/mlp_window.h
core/model/mlp_window.c
tests/test_mlp_window.c
```

Implemented functions:

```text
tkm_mlp_window_init
tkm_mlp_window_forward
tkm_mlp_window_train_mse
```

This is CPU inference plus one-sample MSE training. Full batching, optimizers, and CUDA kernels are not implemented yet.

The layer factory can construct fixed-weight MLP windows from INI plus flat params.

The first fixed-weight end-to-end benchmark path is `centerline_mlp`:

```text
centerline observation -> fixed-weight mlp_window -> categorical argmax -> centerline action
```

This proves the config/factory/model/head/action path.

The first trained MLP benchmark path is `centerline_mlp_trained`:

```text
centerline oracle data -> mlp_window_train_mse -> categorical argmax -> centerline action
```

## What `mlp_window` means

`mlp_window` is a simple feed-forward policy/model over a fixed-size input vector.

Shape:

```text
input vector -> linear -> ReLU -> linear -> output logits
```

For a token/control stack:

```text
recent observations/actions -> embed/window vector -> MLP -> action logits -> categorical head -> decoder
```

This is not a transformer. It does not use attention or a KV cache.

## Why this model first

A small MLP is the simplest real learned-model target after lookup policies.

It is useful because:

```text
works with continuous or embedded inputs
has obvious math
is easy to test by hand
is a good CPU baseline before CUDA
fits centerline, Snake, Breakout, and line follower feature vectors
```

It also gives the registered config kind `mlp_window` real core code behind it.

## Proposed ini shape

```ini
[model]
kind = mlp_window
input_dim = 32
hidden_dim = 64
output_dim = 4
w0 = policy_w0
b0 = policy_b0
w1 = policy_w1
b1 = policy_b1
```

With surrounding layers:

```ini
[tokenizer]
kind = raw_continuous

[embedder]
kind = linear

[sequence_layout]
kind = action_only_prediction

[head]
kind = categorical

[decoder]
kind = argmax

[loss]
kind = cross_entropy
```

The layer factory can construct `mlp_window` from this shape. A full project-level config loader that wires every layer from one project INI file is still future work.

## Current tests

`tests/test_mlp_window.c` covers:

```text
one-hidden-layer forward pass
ReLU activation
optional zero biases
bad input handling
one-sample MSE training
```

## How it fits current layers

Current phase-0 path:

```text
project observation -> tokenizer/vectorizer -> embedder -> mlp_window -> categorical head -> decoder -> action
```

Implemented pieces:

```text
tokenizer: int_bins, vq_code
embedder: lookup, linear
model: lookup_policy, mlp_window, linear_policy, nearest_policy, sparse_lookup, Snake planner_oracle
head: categorical
loss: cross_entropy, mse, huber, binary_cross_entropy, weighted_sum
```

Still missing for a real trained MLP:

```text
dataset batching into model inputs
optimizer
cross-entropy training
Snake/Breakout trained-MLP benchmarks
```

## Planned model options

Useful future models:

```text
ngram
rnn
gru
lstm
transformer_decoder
mamba
temporal_conv
decision_transformer
diffusion_policy
```

The next model should probably be either:

```text
ngram: simple token sequence baseline
tiny transformer decoder: closer to the humanoid-token inspiration
```

The MLP is the pragmatic bridge between lookup policies and sequence models.
