# Decoder fallback options

Date: 2026-07-06

## Why this exists

The action scorer can rank valid actions, but it still occasionally chooses a weaker action than the deterministic Snake planner. A decoder fallback is a clean layer seam for this:

```text
model scores -> masked argmax -> decoder fallback -> action
```

This keeps the fallback out of the model. The model still predicts normally, and the decoder can optionally arbitrate using project-specific knowledge.

## Config

```ini
[decoder]
fallback = none
```

or:

```ini
[decoder]
fallback = planner_score
```

`none` is the generic default.

`planner_score` is Snake-specific in current wiring. It compares the model-selected action against `snake_planner_action` using the existing Snake action score and takes the planner action only when that score is better.

## Result

Fresh benchmark before fallback:

```text
snake_action_scorer_config_file_holdout steps=480 score=83 metric=83
snake_action_scorer_config_file_long_holdout steps=1138 score=144 metric=144
```

With `fallback = planner_score`:

```text
snake_action_scorer_config_file_holdout steps=480 score=89 metric=89
snake_action_scorer_config_file_long_holdout steps=1062 score=175 metric=175
```

This is not a pure survival improvement. It terminals earlier in the long run, but collects much more food before doing so. For the current Snake project, this is a better play result.

## Regression floor

The long-horizon floor is now:

```text
steps >= 1000
score >= 170
metric >= 170
```

This preserves a substantial survival requirement while making score the primary signal for whether Snake is actually playing better.

## DAgger round config

`[train] dagger_rounds` is now parsed and wired into the Snake action-scorer config path.

The current Snake project config keeps:

```ini
dagger_rounds = 4
```

I tested `dagger_rounds = 6` with the current margin scorer, survival mask, and planner-score decoder fallback:

```text
dagger_rounds=4 -> snake_action_scorer_config_file_long_holdout steps=1062 score=175 metric=175
dagger_rounds=6 -> snake_action_scorer_config_file_long_holdout steps=1062 score=171 metric=171
```

The option is useful for experiments, but `6` is rejected as the default because it underperforms the current `4`-round setting.

## Best-score fallback experiment

I also tested:

```ini
[decoder]
fallback = best_score
```

This decoder ignores the model-selected action and picks the action with the highest immediate Snake action score.

Result:

```text
fallback=planner_score -> short score=89, long steps=1062, long score=175
fallback=best_score    -> short score=77, long steps=987,  long score=99
```

`best_score` is kept as an experimental option because it is a useful decoder seam, but it is rejected as the Snake project default.

## Action-scorer hidden size

`[model] hidden_dim` is now parsed and routed into the Snake action-scorer config path.

The current Snake project config keeps:

```ini
[model]
kind = action_scorer
hidden_dim = 32
```

I tested `hidden_dim = 64`:

```text
hidden_dim=32 -> short score=89, long steps=1062, long score=175
hidden_dim=64 -> short score=88, long steps=1200, long score=137
```

The wider scorer survives longer but collects much less food. It is useful as a model-capacity option, but it is rejected as the Snake project default.

## Later update

`planner_score` is no longer the strongest Snake project default. It was superseded by `rollout_score`, documented in `52-rollout-score-decoder.md`:

```text
planner_score -> short score=89,  long score=175
rollout_score -> short score=128, long score=190
```
