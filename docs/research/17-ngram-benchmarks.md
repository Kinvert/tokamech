# Ngram Benchmarks

Date: 2026-07-06

## Current implementation status

The `ngram` next-token model is now used in the benchmark harness:

```text
apps/benchmark.c
tests/test_benchmark.c
```

Benchmark functions:

```text
tkm_bench_centerline_ngram
tkm_bench_breakout_ngram
tkm_bench_snake_ngram
```

## Current benchmark output

```text
centerline steps=8 score=1 metric=1
centerline_ngram steps=8 score=1 metric=1
breakout steps=220 score=2 metric=3
breakout_ngram steps=220 score=2 metric=3
snake_lookup steps=3 score=1 metric=1
snake_ngram steps=3 score=1 metric=1
snake_planner steps=96 score=11 metric=11
```

## How the benchmark stream works

The benchmark builds a simple obs/action token stream:

```text
OBS_TOKEN ACTION_TOKEN OBS_TOKEN ACTION_TOKEN ...
```

Action tokens are offset away from observation tokens:

```text
action_token = 192 + action_id
```

At runtime:

```text
current obs token -> ngram predicts next token -> decode action token
```

If the ngram predicts something outside the action-token range, the benchmark uses a project-specific safe default.

## Interpretation

The ngram baseline matches the current lookup baseline on centerline, Breakout, and one-food Snake.

This is expected because these benchmark streams are short and mostly deterministic:

```text
centerline: offset token strongly determines action
breakout: compact relative-position token strongly determines action
snake: compact token is enough for the first food but not long-term play
```

The ngram model is not yet a strong Snake policy. The strong current Snake baseline remains:

```text
snake_planner steps=96 score=11 metric=11
```

## Why this matters

This is the first benchmark evidence that a generic next-token model path can run across all current projects.

It is not config-driven yet, but it proves the pieces can connect:

```text
project oracle data -> obs/action token stream -> ngram train -> ngram runtime -> project action
```

## Current limitations

Not implemented yet:

```text
generic sequence-layout-to-ngram training
typed token masking
EOS/PAD handling
sampling from ngram counts
longer-context ngrams
config-driven benchmark selection
learned model training
```

The next useful step is a project-aware factory that constructs these benchmark stacks from INI layer names instead of hardcoded benchmark functions.
