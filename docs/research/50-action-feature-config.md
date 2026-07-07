# Action feature config

Date: 2026-07-06

## Why this exists

The action scorer should not hardcode one project feature layout.

For Snake, the scorer takes:

```text
state features + one candidate action row -> scalar action score
```

The action row is now configurable:

```ini
[model]
action_features = space
```

Available modes:

```text
basic       safe bit, distance delta, next food row delta, next food col delta
space       basic + reachable-space fraction
food_space  space + immediate eats-food bit
```

`space` remains the default because it preserves the strongest current Snake config behavior.

## TDD result

Added tests for:

```text
TkmLayerConfig.action_features parsing
TkmRunConfig propagation
snake_write_action_food_space_features
```

The first red build failed because `TkmLayerConfig` had no `action_features` field and `TKM_ACTION_FEATURE_FOOD_SPACE` did not exist. The green implementation added the enum, parser, Snake feature writer, and benchmark routing.

## Benchmark result

Baseline before trying `food_space`:

```text
action_features=space -> short score=89, long steps=1062, long score=175
```

Experiment:

```text
action_features=food_space -> short score=83, long steps=1200, long score=139
```

`food_space` survived longer on the long holdout, but collected far less food. It is kept as an interchangeable representation option, but rejected as the Snake project default.

## Current default

`projects/snake/action_scorer.ini` now states the selected mode explicitly:

```ini
[model]
kind = action_scorer
hidden_dim = 32
action_features = space
```

