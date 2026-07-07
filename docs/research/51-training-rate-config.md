# Training rate config

Date: 2026-07-06

## Why this exists

The Snake action-scorer benchmark had hardcoded trainer rates:

```text
margin base learning rate: 0.01
masked-CE base learning rate: 0.03
DAgger learning rate: 0.005
```

Those are now configurable through `[train]`:

```ini
[train]
learning_rate = 0.01
dagger_learning_rate = 0.005
```

If `learning_rate` is omitted, it stays `auto` and the benchmark uses the loss-specific default. `dagger_learning_rate` defaults to `0.005`.

## TDD result

Added tests for:

```text
TkmTrainConfig.learning_rate parsing
TkmTrainConfig.dagger_learning_rate parsing
TkmRunConfig propagation
invalid zero and non-numeric rates
```

The first red build failed because `TkmTrainConfig` had no `learning_rate` or `dagger_learning_rate` fields. The green implementation added decimal parsing, `auto` handling, config propagation, and benchmark routing.

## Benchmark sweep

Accepted baseline:

```text
learning_rate=0.01, dagger_learning_rate=0.005 -> short score=89, long score=175
```

Tested:

```text
learning_rate=0.02,  dagger_learning_rate=0.005 -> short score=89, long score=175
learning_rate=0.005, dagger_learning_rate=0.005 -> short score=89, long score=175
learning_rate=0.01,  dagger_learning_rate=0.01  -> short score=89, long score=175
learning_rate=1.0,   dagger_learning_rate=1.0   -> short score=89, long score=175
```

The current margin action-scorer stack is insensitive to this sweep on the config-file benchmark. The likely reason is that the seeded action features plus planner-score decoder fallback dominate this particular metric.

The rates remain useful layer knobs, but they are not a current Snake default-improvement lever.

