# Core INI File Loader

Date: 2026-07-06

## What changed

The INI parser now has a reusable file-loading entry point:

```text
tkm_ini_parse_file
```

It lives in:

```text
core/config/ini.h
core/config/ini.c
```

## Why this matters

Project configs should live in project folders, but file IO should not be reimplemented by every app or benchmark.

The intended path is now:

```text
project/*.ini file
tkm_ini_parse_file
layer config / train config parser
project runner or benchmark
```

That keeps config loading boring and centralized.

## Current use

Snake's action-scorer benchmark now loads:

```text
projects/snake/action_scorer.ini
```

through `tkm_ini_parse_file`, then routes the parsed train loss through the action-scorer trainer dispatch.

Current benchmark result:

```text
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
```

## Limits

The loader uses a bounded file size:

```text
TKM_INI_MAX_FILE_BYTES = 4096
```

That is enough for current small project configs. If future configs need larger sections for arrays or model weights, those should move into params/checkpoint files instead of bloating the INI.

## Tests

Coverage was added in:

```text
tests/test_ini.c
```

The tests verify:

```text
valid INI files parse
missing files fail
null ini or path fails
```
