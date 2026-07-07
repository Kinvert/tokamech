# Cross-Project Action Scorer Configs

Date: 2026-07-06

## What changed

All three current projects now have project-local action-scorer config files:

```text
projects/centerline/action_scorer.ini
projects/breakout/action_scorer.ini
projects/snake/action_scorer.ini
```

Centerline and Breakout use:

```ini
[project]
name = centerline

[model]
kind = action_scorer
```

Snake also selects its training objective:

```ini
[project]
name = snake

[model]
kind = action_scorer

[train]
loss = masked_cross_entropy
```

## Benchmark rows

```text
centerline_action_scorer_config_file steps=8 score=1 metric=1
breakout_action_scorer_config_file steps=220 score=2 metric=3
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
```

## Why this matters

The action scorer is no longer only a Snake-specific or hard-coded benchmark trick.

Each current project now has a local INI file that validates:

```text
project folder owns config
core INI loader reads config
project config validates project name
layer config validates model kind
benchmark dispatches to the selected model path
```

That is still not a full generic project runner, but it is the same seam repeated across all current envs.

Configs are project-specific. A Centerline benchmark rejects a Breakout config file even if the model kind is valid.

## Current limit

Centerline and Breakout currently use deterministic scorer weights and do not have train-loss config rows.

Snake is the stronger proof because its config also selects `[train] loss = masked_cross_entropy` and runs a held-out learned policy benchmark.

Project identity validation is covered separately in `45-project-config-validation.md`.
