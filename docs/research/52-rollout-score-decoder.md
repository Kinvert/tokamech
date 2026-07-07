# Rollout-score decoder fallback

Date: 2026-07-06

## Why this exists

The Snake action scorer can rank candidate actions, but one-step scoring still misses traps. A decoder fallback can use project-specific simulation after the model proposes an action:

```text
model scores -> masked argmax -> rollout_score decoder fallback -> action
```

`rollout_score` evaluates each first action by simulating that action, then following the existing Snake planner for a short horizon. The score favors food collected and survival. Ties preserve the model-selected action.

This is a decoder/planner seam, not a model change. It is intentionally project-specific in the current wiring.

## Config

```ini
[decoder]
fallback = rollout_score
```

The previous Snake default was:

```ini
fallback = planner_score
```

## TDD result

Added tests for:

```text
rollout_score string parsing
TkmRunConfig propagation
snake_filter_rollout_score_action avoiding a dead-end move
```

The first red build failed because `TKM_DECODER_FALLBACK_ROLLOUT_SCORE` did not exist. The green implementation added the parser enum, Snake rollout scorer, and benchmark routing.

## Benchmark result

Previous accepted default:

```text
fallback=planner_score -> short score=89, long steps=1062, long score=175
```

New accepted default:

```text
fallback=rollout_score -> short score=128, long steps=1200, long score=190
```

The long holdout also survives all five starts for the full 240-step horizon:

```text
5 starts * 240 steps = 1200 steps
```

This is the strongest current Snake project config result.

## Regression floor

The Snake config-file benchmark floor is now:

```text
short score >= 125
long steps >= 1200
long score >= 245
```

These floors protect the horizon-32 improvement without requiring the exact current `131/255` result.

## Rollout horizon sweep

`[decoder] rollout_horizon` is now parsed and routed into the Snake action-scorer config path.

Tested horizons:

```text
rollout_horizon=8  -> short score=92,  long steps=1071, long score=157
rollout_horizon=16 -> short score=128, long steps=1200, long score=190
rollout_horizon=32 -> short score=131, long steps=1200, long score=255
rollout_horizon=64 -> short score=102, long steps=1200, long score=174
```

`32` is the best tested value. The project-local Snake config now uses:

Expanded sweep around `32`:

```text
rollout_horizon=24 -> short score=125, long steps=1195, long score=239
rollout_horizon=28 -> short score=129, long steps=1200, long score=237
rollout_horizon=36 -> short score=112, long steps=1200, long score=221
rollout_horizon=40 -> short score=102, long steps=1200, long score=170
rollout_horizon=48 -> short score=101, long steps=1198, long score=189
```

The expanded sweep still supports `32` as the default.

```ini
[decoder]
fallback = rollout_score
rollout_horizon = 32
```

## Rollout score mode experiment

`[decoder] rollout_score_mode` is now parsed and routed into the Snake action-scorer config path.

Available modes:

```text
food_steps      current default; food collected dominates, survival steps break ties
space_distance  experimental; also rewards final reachable space and food distance
```

Benchmark:

```text
rollout_score_mode=food_steps     -> short score=131, long steps=1200, long score=255
rollout_score_mode=space_distance -> short score=65,  long steps=1200, long score=115
```

`space_distance` is kept as an interchangeable decoder scoring option, but it is rejected as the Snake project default.
