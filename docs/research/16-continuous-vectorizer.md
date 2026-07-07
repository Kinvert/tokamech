# Continuous Vectorizer

Date: 2026-07-06

## Current implementation status

The first continuous vectorizer is implemented in core:

```text
core/vectorizer/continuous.h
core/vectorizer/continuous.c
tests/test_continuous_vectorizer.c
```

Implemented functions:

```text
tkm_continuous_vectorizer_init_raw
tkm_continuous_vectorizer_init_standardize
tkm_continuous_vectorizer_apply
```

The layer-kind registry now recognizes:

```ini
[tokenizer]
kind = normalized_continuous
```

The layer factory can construct standardized vectorizers from INI plus params:

```ini
[tokenizer]
kind = normalized_continuous
dim = 2
mean = qti_mean
std = qti_std
```

```text
qti_mean = 10.0 0.0
qti_std = 2.0 4.0
```

The registry also already recognizes:

```ini
[tokenizer]
kind = raw_continuous
```

## What this layer does

This layer handles continuous float vectors.

Raw mode:

```text
input -> output
```

Standardized mode:

```text
output[i] = (input[i] - mean[i]) / std[i]
```

The implementation stores `1 / std[i]` so runtime uses multiplication.

## Why this matters

Not every project should be forced through integer tokens.

Useful examples:

```text
line follower QTI readings normalized to floats
Snake head/food/body feature vector
Breakout paddle/ball positions
robot joint angles and velocities
```

These can feed:

```text
continuous vectorizer -> linear embedder -> mlp_window
```

or later:

```text
continuous vectorizer -> learned VQ -> token sequence model
```

## Proposed ini shape

Raw continuous:

```ini
[tokenizer]
kind = raw_continuous
dim = 11
```

Standardized continuous:

```ini
[tokenizer]
kind = normalized_continuous
dim = 11
stats = project_stats
```

With a model:

```ini
[embedder]
kind = linear
input_dim = 11
output_dim = 64

[model]
kind = mlp_window
```

## Current tests

`tests/test_continuous_vectorizer.c` covers:

```text
raw vector copy
per-dimension standardization
bad input handling
invalid zero/negative std rejection
```

`tests/test_layer_config.c` covers:

```text
"normalized_continuous" -> TKM_TOKENIZER_NORMALIZED_CONTINUOUS
```

## Current limitations

Not implemented yet:

```text
file-loaded mean/std stats
min/max scaling
clipping
running mean/std estimation
batch vectorization
project factory wiring
benchmarks comparing raw_continuous vs int_bins
```

## Architecture rule

Project code still builds the raw float vector explicitly.

Good:

```c
features[0] = qti[0];
features[1] = qti[1];
features[2] = left_motor;
features[3] = right_motor;
```

Core vectorizer:

```text
float input[dim] -> float output[dim]
```

Bad:

```text
core/vectorizer knows what QTI or snake food means
```

The vectorizer should only transform flat numeric arrays.
