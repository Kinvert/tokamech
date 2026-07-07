# Flat Parameter Store

Date: 2026-07-06

## Current implementation status

The first flat parameter store is implemented in core:

```text
core/params/params.h
core/params/params.c
tests/test_params.c
```

Implemented functions:

```text
tkm_params_init
tkm_params_add_array
tkm_params_get_array
tkm_params_parse_text
```

## Text format

The current parser supports simple named float arrays:

```text
codebook = 0.0, 1.0, -2.5
bias = 3.0 4.0
```

Comments are supported:

```text
# comment
; comment
```

Commas are treated like whitespace.

## Why this matters

The scalar layer factory can build simple layers from INI:

```text
int_bins
raw_continuous
ngram
```

The params-backed layer factory can now build:

```text
vq_code
lookup embedder
linear embedder
mlp_window
normalized_continuous
```

The flat parameter store is the bridge for those layers.

## Current tests

`tests/test_params.c` covers:

```text
add named float array
retrieve named float array
parse text with comments
parse comma-separated and whitespace-separated floats
reject duplicate names
reject malformed values
bad input handling
```

## Factory integration

Current factory functions that take both config and params:

```text
tkm_layer_factory_init_vq_code(ini, params, out)
tkm_layer_factory_init_lookup_embedder(ini, params, out)
tkm_layer_factory_init_linear_embedder(ini, params, out)
tkm_layer_factory_init_mlp_window(ini, params, out)
tkm_layer_factory_init_normalized_continuous(ini, params, out)
```

Example INI:

```ini
[tokenizer]
kind = vq_code
dim = 2
codes = 3
codebook = snake_vq_codebook
```

Example params:

```text
snake_vq_codebook = 0.0 0.0 1.0 0.0 0.0 1.0
```

The factory checks that the named parameter array has the required count.

## Current limitations

This is not a final checkpoint format.

Not implemented yet:

```text
binary parameter files
file I/O helper
checksums
dtype metadata
shape metadata beyond what INI declares
parameter saving
CUDA upload
```

For now, it is enough to keep codebook/table/weight loading explicit and testable.
