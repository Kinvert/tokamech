# Snake Action Scorer INI Benchmark

Date: 2026-07-06

## What changed

The Snake action-scorer benchmark can now select the training objective from INI text:

```ini
[train]
loss = margin
```

or:

```ini
[train]
loss = masked_cross_entropy
```

The public benchmark entry point is:

```text
tkm_bench_snake_action_scorer_config_holdout
tkm_bench_snake_action_scorer_config_file_holdout
```

The config path validates both:

```text
[model] kind = action_scorer
[train] loss = margin | masked_cross_entropy
```

Configs with a different model kind fail cleanly instead of silently running the action scorer anyway.

## Why this matters

This makes the config seam real for one complete benchmark path.

Before this, the repo could parse `train.loss`, and the core action-scorer trainer could dispatch by `TkmTrainLossKind`, but the benchmark still exposed only hard-coded rows. Now the path is:

```text
INI text -> TkmTrainConfig -> TkmTrainLossKind -> action scorer trainer dispatch -> Snake holdout benchmark
```

That is closer to the intended project architecture where users can swap methods in an INI file without rewriting project feature code.

## Result

```text
snake_action_scorer_config_margin_holdout steps=480 score=58 metric=58
snake_action_scorer_config_ce_holdout steps=480 score=58 metric=58
```

Both config-selected losses match the direct benchmark rows.

## Project-local INI

The first project-local Snake config lives at:

```text
projects/snake/action_scorer.ini
```

It currently selects:

```ini
[model]
kind = action_scorer

[train]
loss = masked_cross_entropy
```

The benchmark can now load this file directly:

```text
tkm_bench_snake_action_scorer_config_file_holdout("projects/snake/action_scorer.ini")
```

The file is loaded through the reusable core helper:

```text
tkm_ini_parse_file
```

That produces:

```text
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
```

## Current limit

This is still specific to the Snake action-scorer benchmark.

It does not yet mean every project can be fully trained from one INI file. The next architectural step is a small project-level train runner that accepts:

```text
project callbacks
layer config
train config
benchmark/evaluation callback
```

and owns the common training loop.

The file loader itself is now shared core config code. See `42-core-ini-file-loader.md`.
