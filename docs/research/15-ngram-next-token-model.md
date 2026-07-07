# Ngram Next-Token Model

Date: 2026-07-06

## Current implementation status

The first ngram model is implemented in core:

```text
core/model/ngram.h
core/model/ngram.c
tests/test_ngram_model.c
```

Implemented functions:

```text
tkm_ngram_init
tkm_ngram_train_sequence
tkm_ngram_predict_next
```

The layer-kind registry now recognizes:

```ini
[model]
kind = ngram
```

## What is implemented

The current model is a bigram count model:

```text
previous token -> most common next token
```

Training counts adjacent token pairs:

```text
token[t - 1], token[t]
```

Prediction chooses the most common next token for the previous token. Ties choose the lowest token ID. Unknown contexts return token `0`.

## Why this matters

This is the simplest real next-token model in the repo.

It is useful because:

```text
directly matches the next-token framing
is deterministic
is easy to inspect
requires no neural net training
can test sequence layout output before transformers exist
gives a baseline between lookup_policy and transformer_decoder
```

`lookup_policy` maps an observation token to an action. `ngram` maps a previous stream token to the next stream token.

That distinction matters for studying papers that treat control as sequence modeling.

## Proposed ini shape

```ini
[model]
kind = ngram
vocab_size = 128
order = 2
```

Only order 2 is implemented right now.

## Current tests

`tests/test_ngram_model.c` covers:

```text
bigram count training
most-common next-token prediction
lowest-token tie behavior
unknown-context fallback
bad input handling
```

`tests/test_layer_config.c` covers:

```text
"ngram" -> TKM_MODEL_NGRAM
```

## Current limitations

Not implemented yet:

```text
order > 2
smoothing
sampling from counts
log probabilities
typed sequence item training
EOS/PAD masking
batch training
config-driven construction
benchmarks using sequence layout output
```

## Relationship to transformer work

The ngram model is not intended to compete with transformers.

It gives a tiny, auditable next-token baseline:

```text
if ngram fails, inspect the token stream first
if ngram works, a transformer has a sane baseline to beat
```

This is valuable for line follower, centerline, and Snake because it lets the repo test the sequence formulation before CUDA attention or KV cache complexity enters the codebase.
