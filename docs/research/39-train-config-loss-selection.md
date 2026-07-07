# Train Config Loss Selection

Date: 2026-07-06

## What changed

The repo now has a small training config parser:

```text
core/config/train_config.h
core/config/train_config.c
```

It reads:

```ini
[train]
loss = masked_cross_entropy
```

Supported values:

```text
cross_entropy
masked_cross_entropy
mse
margin
```

The default is:

```text
cross_entropy
```

## Why this matters

This separates architecture selection from training selection.

`core/config/layer_config` answers questions like:

```text
which tokenizer?
which model?
which decoder?
```

`core/config/train_config` answers:

```text
which training loss or objective?
```

That distinction matters for the action scorer. The same model can now be paired with either:

```text
margin
masked_cross_entropy
```

The benchmark already proves both train the current Snake scorer to the same held-out result:

```text
snake_action_scorer_holdout steps=480 score=58 metric=58
snake_action_scorer_ce_holdout steps=480 score=58 metric=58
```

## Current status

The parser support is now connected to a reusable action-scorer trainer dispatch:

```text
core/train/action_scorer_trainer.h
core/train/action_scorer_trainer.c
```

That dispatch can route:

```text
TKM_TRAIN_LOSS_MARGIN
TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY
```

See `40-action-scorer-trainer-dispatch.md`.

## Current limit

This still does not route a complete project INI into a full train loop. Benchmarks can call the dispatch directly, but there is not yet a reusable project-level trainer that owns rollout collection, DAgger rounds, held-out splits, or checkpoint writing.

## Tests

Coverage was added in:

```text
tests/test_train_config.c
```

The tests verify:

```text
train.loss reads from INI
default loss is cross_entropy
margin and masked_cross_entropy parse successfully
unknown loss strings fail
null inputs fail
```
