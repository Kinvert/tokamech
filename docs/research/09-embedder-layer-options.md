# Embedder Layer Options

Date: 2026-07-06

## Current implementation status

The first embedder module is implemented in core:

```text
core/embedder/embedder.h
core/embedder/embedder.c
tests/test_embedder.c
```

Implemented types:

```text
TkmLookupEmbedder
TkmLinearEmbedder
```

Implemented functions:

```text
tkm_lookup_embedder_init
tkm_lookup_embedder_encode
tkm_lookup_embedder_encode_batch
tkm_linear_embedder_init
tkm_linear_embedder_encode
```

This is inference math only. Training/updating embedding weights is not implemented yet.

The layer factory can construct lookup and linear embedders from INI plus flat params.

## Tokenizer versus embedder

Tokenizer:

```text
raw observation/action data -> token ID or continuous vector
```

Embedder:

```text
token ID or continuous vector -> model-width float vector
```

Example:

```text
int_bins: sensor value 573 -> token 17
lookup_embedder: token 17 -> [0.12, -0.03, 0.88, ...]
```

For continuous inputs:

```text
raw_continuous: [head_x, head_y, food_x, food_y]
linear_embedder: that vector -> model-width float vector
```

## Current embedder options

Lookup embedding:

```ini
[embedder]
kind = lookup
tokens = 128
dim = 64
table = token_embedding_table
```

Good for:

```text
discrete token IDs
integer bins
packed tokens
VQ codes
actions as enum tokens
special tokens like BOS/EOS/PAD
```

Linear embedding:

```ini
[embedder]
kind = linear
input_dim = 11
output_dim = 64
weights = linear_embedding_w
bias = linear_embedding_b
```

Good for:

```text
continuous observations
small hand-built feature vectors
normalized robotics sensor vectors
game state vectors
```

## Why these first

These two embedders are enough to connect the current tokenizers to future models.

```text
int_bins -> lookup
packed_token -> lookup
vq_code -> lookup
raw_continuous -> linear
```

They are deterministic, flat-buffer C, easy to test, and do not require a trainer before the data path is clear.

## Current tests

`tests/test_embedder.c` covers:

```text
lookup token -> vector copy
lookup batch encode
linear projection with bias
bad input handling
```

## Planned embedder options

Useful future embedders:

```text
typed_lookup              token embedding + token type embedding
learned_position          learned position vector added to token embedding
sinusoidal_position       deterministic transformer position encoding
rope                      rotary position embedding for attention
alibi                     attention bias rather than direct vector embedding
fourier_features          scalar -> sin/cos feature vector
image_patch_linear        image patch -> model vector
vq_lookup                 VQ code -> code embedding table
```

## Architecture rule

Core embedders should not know project semantics.

Good:

```text
uint16_t token -> float embedding[dim]
float input[input_dim] -> float embedding[output_dim]
```

Bad:

```text
QTI sensor -> special line follower embedding
snake food -> special snake embedding
breakout paddle -> special breakout embedding
```

Projects can choose how to build raw observations. Core should provide reusable math modules that operate on flat arrays.
