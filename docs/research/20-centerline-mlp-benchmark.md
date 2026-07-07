# Centerline MLP Benchmark

Date: 2026-07-06

## What changed

`centerline_mlp` is the first benchmark that drives an environment through a factory-built `mlp_window` model.

The path is:

```text
INI model config -> flat params -> layer factory -> mlp_window -> categorical argmax -> action command
```

It is intentionally fixed-weight. This is not training yet.

## Why centerline first

Centerline is hand-checkable:

```text
offset < 0 -> move right
offset = 0 -> stay
offset > 0 -> move left
```

That makes it a good test target for proving model wiring without hiding bugs behind a complex environment.

## Model shape

The benchmark uses:

```text
input_dim = 1
hidden_dim = 3
output_dim = 3
```

Hidden units:

```text
h0 = relu(offset)
h1 = relu(-offset)
h2 = relu(1)
```

Output logits map to Centerline actions:

```text
left  = 2 * h0
stay  = 1 * h2
right = 2 * h1
```

This makes positive offsets choose `LEFT`, zero choose `STAY`, and negative offsets choose `RIGHT`.

## Current benchmark result

Command:

```sh
make benchmark
```

Output line:

```text
centerline_mlp steps=8 score=1 metric=1
```

## What this proves

This proves:

```text
mlp_window can be constructed from INI plus named params
mlp_window can be used as a policy model in a real project loop
categorical argmax head can convert logits to action ids
the existing benchmark/test loop can compare lookup, ngram, and MLP paths
```

## What this does not prove

This does not prove:

```text
MLP training works
Snake benefits from MLP yet
Breakout benefits from MLP yet
CUDA kernels exist
```

Those remain separate TDD steps.
