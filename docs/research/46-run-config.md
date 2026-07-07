# Run Config

Date: 2026-07-06

## What changed

The repo now has a combined run config parser:

```text
core/config/run_config.h
core/config/run_config.c
```

It combines:

```text
TkmProjectConfig
TkmLayerConfig
TkmTrainConfig
```

into:

```text
TkmRunConfig
```

## API

```text
tkm_run_config_from_ini
tkm_run_config_from_file
```

## Why this matters

The config path is now closer to the intended architecture:

```text
project-local INI
core INI file loader
run config parser
project runner or benchmark
```

Before this, benchmark wrappers parsed project, layer, and train config separately. That worked, but it made every future runner likely to duplicate the same validation sequence.

Now a runner can ask for one config object and then decide what to execute.

## Current use

The action-scorer config-file benchmarks now validate through `TkmRunConfig`.

Current rows:

```text
centerline_action_scorer_config_file steps=8 score=1 metric=1
breakout_action_scorer_config_file steps=220 score=2 metric=3
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
```

## Current limit

This is still a parser and validation object, not a full generic runner.

The next step is a small dispatch layer that accepts `TkmRunConfig` and chooses a project/model benchmark or training path.

## Tests

Coverage was added in:

```text
tests/test_run_config.c
```

The tests verify:

```text
project/model/train sections parse together
project-local INI files parse into TkmRunConfig
missing project or missing files fail
null inputs fail
```
