# Project Config Validation

Date: 2026-07-06

## What changed

The repo now has a small project config parser:

```text
core/config/project_config.h
core/config/project_config.c
```

It reads:

```ini
[project]
name = snake
```

Supported project names:

```text
centerline
breakout
snake
```

## Why this matters

Project-local config files should identify the project they belong to.

Before this, a config file could select a valid model kind but still be used by the wrong project benchmark. Now benchmark wrappers validate:

```text
expected project name
model kind
train loss where relevant
```

That makes the config seam safer before a full project-level runner exists.

## Current project configs

```text
projects/centerline/action_scorer.ini
projects/breakout/action_scorer.ini
projects/snake/action_scorer.ini
```

Each file includes `[project] name = ...`.

## Tests

Coverage was added in:

```text
tests/test_project_config.c
tests/test_benchmark.c
```

The tests verify:

```text
project names parse
missing or unknown project names fail
valid project-local action-scorer configs run
wrong-project config files fail cleanly
```

## Current limit

This validates project identity, but it still does not create a generic runner that can choose and execute any project by config alone. It is the validation layer that such a runner should use.

The next layer is `TkmRunConfig`, covered in `46-run-config.md`.
