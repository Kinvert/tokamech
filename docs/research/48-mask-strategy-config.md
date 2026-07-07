# Mask strategy config seam

Date: 2026-07-06

## Why this exists

Action masks are a layer in this repo. They decide which actions a model may choose from during training and inference.

For Snake, an immediate-collision mask was not enough after old-tail moves became legal. The model could choose actions that were technically legal for one step but entered tiny dead ends.

## Config

Mask strategy is now parsed from INI:

```ini
[mask]
strategy = immediate
```

or:

```ini
[mask]
strategy = survival
```

Default is `immediate`, because that is the simplest generic mask: only hide actions that terminate immediately.

Snake's project config uses:

```ini
[train]
loss = margin

[mask]
strategy = survival
```

## Implemented strategies

`immediate`

Only rejects actions that terminal on the next step.

`survival`

Rejects immediate terminal actions and, when possible, rejects actions whose post-step reachable space is too small for the snake length. If every action is cramped, it falls back to the immediate-valid mask so the action head is not given an all-zero mask.

## Current wiring

`TkmRunConfig` now includes `TkmMaskConfig`.

The Snake action-scorer benchmark config path reads `[mask]` and routes training plus inference through the selected mask strategy. The project-local `projects/snake/action_scorer.ini` therefore changes real behavior, not just parsed metadata.

## Benchmark contract

`tests/test_benchmark.c` now checks that Snake's survival mask config beats the default immediate mask for the action-scorer CE holdout.

The project-local Snake config file now uses margin loss plus survival masking because that is the strongest current verified action-scorer option.

The project-local Snake config file must keep at least an `80` score floor. Current verified score is `83`.

Masked cross entropy with survival masking remains available and currently verifies at `68`, but it is no longer the default project-local Snake config.

## Training horizon config

`[train] rollout_steps` is now parsed into `TkmTrainConfig` and wired into the Snake action-scorer config training path.

The current Snake project config is explicit:

```ini
[train]
loss = margin
rollout_steps = 96
```

I tested longer rollout training and rejected it as the default:

```text
rollout_steps=128 -> short score=49, long score=92
rollout_steps=240 -> short score=67, long score=116
rollout_steps=96  -> short score=83, long score=144
```

Longer training improved raw survival at `240`, but the learned scorer became too conservative and collected much less food. For this simple action-scorer setup, `96` is the better current tradeoff.

## Long-horizon benchmark

The benchmark now also includes:

```text
snake_action_scorer_config_file_long_holdout
```

This runs the project-local Snake action-scorer config for five holdout starts at `240` steps each. The regression floor is:

```text
steps >= 1100
score >= 130
metric >= 130
```

Current verified row:

```text
snake_action_scorer_config_file_long_holdout steps=1062 score=175 metric=175
```

I tested a stricter low-space runtime filter. It did not improve the long benchmark: it kept the same `1138` steps and reduced food to `138`, so it was not kept in the default path.

The later decoder fallback work is covered in `docs/research/49-decoder-fallback.md`.
