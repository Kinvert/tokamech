# Action Scorer Cross-Project Validation

Date: 2026-07-06

## Why this matters

The action scorer should not be a Snake-only trick.

The architecture goal is interchangeable layers that work across simple control projects. This benchmark slice validates the same core scorer API on all current environments:

```text
Centerline
Breakout
Snake
```

## Results

```text
centerline_action_scorer steps=8 score=1 metric=1
breakout_action_scorer steps=220 score=2 metric=3
snake_action_scorer_holdout steps=480 score=58 metric=58
snake_action_scorer_ce_holdout steps=480 score=58 metric=58
```

The scorer matches the existing Centerline and Breakout baselines and remains the best current learned Snake policy.

## Centerline mapping

Centerline uses a one-dimensional state:

```text
state = current offset
```

Each candidate action writes one action feature:

```text
1 - abs(next_offset) / scale
```

The scorer chooses the action whose next state is closest to zero.

## Breakout mapping

Breakout uses the existing paddle/ball state from the env.

Each candidate action writes one action feature:

```text
1 - abs(ball_center - next_paddle_center) / screen_width
```

The scorer chooses the paddle move that best tracks the ball.

## Snake mapping

Snake uses richer state and candidate-action features:

```text
state = board, head, food, direction, length
action = validity, food-distance delta, next food vector, reachable-space fraction
```

The reachable-space feature is important for avoiding moves that are valid now but trap the snake later.

## Lesson

The scorer layer is now a real cross-project option.

The model is not enough by itself. Each project still owns clear feature-writing code, and the scorer only consumes those explicit vectors. This keeps the system understandable while still allowing the core layer to be shared.
