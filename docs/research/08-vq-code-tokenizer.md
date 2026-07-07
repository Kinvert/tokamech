# VQ Code Tokenizer

Date: 2026-07-06

## Current implementation status

The first VQ tokenizer module is implemented in core:

```text
core/tokenizer/vq_code.h
core/tokenizer/vq_code.c
tests/test_vq_code.c
```

Implemented functions:

```text
tkm_vq_code_init
tkm_vq_code_encode
tkm_vq_code_decode
tkm_vq_code_encode_batch
```

This is fixed-codebook vector quantization. Learned codebooks are not implemented yet.

The layer factory can construct fixed VQ tokenizers from INI plus flat params:

```ini
[tokenizer]
kind = vq_code
dim = 2
codes = 3
codebook = toy_codebook
```

```text
toy_codebook = 0.0 0.0 1.0 0.0 0.0 1.0
```

## What VQ means here

VQ means vector quantization.

A continuous vector is mapped to the nearest entry in a codebook:

```text
input vector -> nearest codebook vector -> integer code
```

Example:

```text
code 0 = [0.0, 0.0]
code 1 = [1.0, 0.0]
code 2 = [0.0, 1.0]

input = [0.9, 0.1]
nearest = code 1
```

That code can then be treated like a token.

## Why this matters for Tokamech

The repo should not be limited to hand-made integer bins.

Useful tokenization methods should include:

```text
int_bins              scalar -> uniform integer bin
packed_token          several discrete fields -> one integer token
raw_continuous        no discrete token, pass vector forward
vq_code               continuous vector -> nearest codebook token
learned_vq_code       planned, train codebook from data
residual_vq           planned, multiple codebooks for more detail
```

Fixed VQ is a good first step because it is deterministic, testable, and easy to review.

## Proposed ini shape

Fixed codebook:

```ini
[tokenizer]
kind = vq_code
codebook = project_codebook
dim = 4
codes = 32
```

Later learned codebook:

```ini
[tokenizer]
kind = learned_vq_code
dim = 16
codes = 256
commitment_weight = 0.25
```

Residual VQ later:

```ini
[tokenizer]
kind = residual_vq
dim = 16
codes = 256
levels = 4
```

## Ownership boundary

Projects still own raw observations.

Example line follower code should stay explicit:

```c
obs[0] = qti[0];
obs[1] = qti[1];
obs[2] = qti[2];
obs[3] = qti[3];
```

Core VQ only receives a flat vector:

```text
float vector[dim] -> uint16_t code
```

Core should not know what QTI sensors, snake bodies, bricks, or paddles are.

## Current tests

`tests/test_vq_code.c` covers:

```text
nearest-code encode
decode code vector
batch encode over flat vectors
bad input handling
```

## Future work

Useful next steps:

```text
k-means codebook fitting for offline datasets
EMA VQ codebook updates
commitment and codebook losses
dead-code replacement
residual VQ
project ini loading for fixed codebooks
benchmarks comparing int_bins vs vq_code on current envs
```

The first benchmark should be conservative: prove VQ can reproduce a simple hand-built codebook on centerline or Snake before using it as a learned representation.
