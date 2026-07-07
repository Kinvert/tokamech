# Layer Factory

Date: 2026-07-06

## Current implementation status

The first config-driven layer factory is implemented in core:

```text
core/config/layer_factory.h
core/config/layer_factory.c
tests/test_layer_factory.c
```

Implemented constructors:

```text
tkm_layer_factory_init_int_bins
tkm_layer_factory_init_raw_continuous
tkm_layer_factory_init_normalized_continuous
tkm_layer_factory_init_ngram
tkm_layer_factory_init_vq_code
tkm_layer_factory_init_lookup_embedder
tkm_layer_factory_init_linear_embedder
tkm_layer_factory_init_mlp_window
tkm_layer_factory_init_action_scorer
```

The ngram benchmark path now uses the factory to construct the ngram model from INI text.

## Supported scalar-only construction

Integer bins:

```ini
[tokenizer]
kind = int_bins
min = -4
max = 4
```

Raw continuous vectorizer:

```ini
[tokenizer]
kind = raw_continuous
dim = 11
```

Ngram model:

```ini
[model]
kind = ngram
vocab_size = 196
```

Fixed VQ codebook:

```ini
[tokenizer]
kind = vq_code
dim = 2
codes = 3
codebook = toy_codebook
```

With params:

```text
toy_codebook = 0.0 0.0 1.0 0.0 0.0 1.0
```

Normalized continuous vectorizer:

```ini
[tokenizer]
kind = normalized_continuous
dim = 2
mean = qti_mean
std = qti_std
```

With params:

```text
qti_mean = 10.0 0.0
qti_std = 2.0 4.0
```

Lookup embedder:

```ini
[embedder]
kind = lookup
tokens = 3
dim = 2
table = token_table
```

With params:

```text
token_table = 1.0 2.0 3.0 4.0 5.0 6.0
```

Linear embedder:

```ini
[embedder]
kind = linear
input_dim = 2
output_dim = 3
weights = linear_w
bias = linear_b
```

MLP window:

```ini
[model]
kind = mlp_window
input_dim = 2
hidden_dim = 3
output_dim = 2
w0 = mlp_w0
b0 = mlp_b0
w1 = mlp_w1
b1 = mlp_b1
```

Action scorer:

```ini
[model]
kind = action_scorer
state_dim = 1
action_dim = 2
hidden_dim = 1
w0 = scorer_w0
b0 = scorer_b0
w1 = scorer_w1
b1 = scorer_b1
```

With params:

```text
scorer_w0 = 0.0 1.0 1.0
scorer_b0 = 0.0
scorer_w1 = 1.0
scorer_b1 = 0.0
```

## What is still deliberately narrow

Some layers still require work beyond flat parameter loading:

```text
transformer_decoder needs many parameter tensors and runtime kernels
learned_vq_code needs codebook training
residual_vq needs multi-level codebook handling
optimizer/training stacks need mutable parameter updates
```

The factory should not fake those. It should return `TKM_ERR` until the required runtime/training pieces exist.

## Current benchmark evidence

Current ngram benchmark rows are built through factory-backed model initialization:

```text
centerline_ngram steps=8 score=1 metric=1
breakout_ngram steps=220 score=2 metric=3
snake_ngram steps=3 score=1 metric=1
```

## Current tests

`tests/test_layer_factory.c` covers:

```text
construct int_bins from INI
construct raw_continuous vectorizer from INI
construct ngram from INI
construct vq_code from INI plus params
construct normalized_continuous from INI plus params
construct lookup embedder from INI plus params
construct linear embedder from INI plus params
construct MLP window from INI plus params
construct action scorer from INI plus params
reject missing required scalar fields
reject wrong layer kinds
reject missing VQ codebook params
reject wrong VQ codebook shape
reject missing or wrong-shaped normalization params
reject wrong-shaped embedder and MLP params
reject wrong-shaped action scorer params
```

## Next useful step

Add file loading for flat params:

```text
read params text from disk
load project params beside project ini
construct full stack from project files
```

After that, the factory can construct more of the stack from project `.ini` files without hardcoded benchmark setup.
