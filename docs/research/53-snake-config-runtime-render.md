# Snake config runtime and render path

Date: 2026-07-06

## Why this exists

The strongest Snake behavior was available in benchmarks through:

```text
projects/snake/action_scorer.ini
fallback = rollout_score
rollout_horizon = 32
```

But the Raylib renderer still showed only planner and lookup modes. That meant visual testing did not exercise the strongest configured policy.

## Change

Added project-local runtime helpers:

```c
uint8_t snake_decode_action(const SnakeEnv* env, const TkmDecoderConfig* decoder, uint8_t suggested_action);

int snake_run_decoder_config_policy(
    SnakeEnv* env,
    const TkmDecoderConfig* decoder,
    uint32_t max_steps,
    SnakeRolloutStats* stats
);
```

These live in `projects/snake/`, not `core/`, because Snake rollout scoring is project-specific. Core remains project-agnostic.

`apps/snake_render.c` now loads:

```text
projects/snake/action_scorer.ini
```

and defaults to `config` policy mode. Pressing `T` cycles:

```text
config -> planner -> lookup
```

## TDD result

The red test required:

```text
snake_decode_action
snake_run_decoder_config_policy
```

and failed to build until those project-local runtime helpers existed.

The headless test verifies that the configured decoder policy can run 96 steps, collect at least 12 food, and avoid terminal.

## Benchmark status

This does not change benchmark policy behavior. It connects the renderer to the already accepted configured policy:

```text
snake_action_scorer_config_file_holdout steps=480 score=131 metric=131
snake_action_scorer_config_file_long_holdout steps=1200 score=255 metric=255
```

