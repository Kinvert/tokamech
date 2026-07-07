# Snake tail moves and survival-aware masks

Date: 2026-07-06

## Context

The Snake env was using an immediate collision rule: any move into a body cell terminaled before the tail advanced. That is too strict for standard Snake. A non-growing move into the current tail cell should be legal because the tail leaves that cell during the same step.

This matters for learned policies because the action mask defines what the model is allowed to choose. If the mask is too strict, good escape moves are hidden. If the mask is too loose, the model can choose immediate-valid moves that enter tiny dead ends.

## Change kept

`snake_step_env` now allows a non-growing move into the old tail cell.

`snake_write_action_mask` now uses a two-stage rule:

1. Compute immediate-valid actions.
2. Compute survival-safe actions with the same reachable-space check used by the planner.
3. If at least one survival-safe action exists, expose only those.
4. If none exist, fall back to immediate-valid actions so the mask never hides every possible move.

This keeps the model/action head interface simple: the mask is still `uint8_t[action_count]`, but it carries a stronger safety contract.

This behavior is also available through the project config seam described in `docs/research/48-mask-strategy-config.md`.

## Change rejected

I tried adding an explicit `moves_into_old_tail` bit to each Snake action-space feature row.

That made the representation larger but did not help the current scorer. It hurt the CE/config path during benchmarks, so it was removed. The better seam for this specific issue is the action mask, not another action feature.

## Benchmark impact

Headless benchmark after the kept change:

```text
centerline_action_scorer_config_file steps=8 score=1 metric=1
breakout_action_scorer_config_file steps=220 score=2 metric=3
snake_action_scorer_holdout steps=480 score=83 metric=83
snake_action_scorer_ce_holdout steps=480 score=68 metric=68
snake_action_scorer_config_margin_holdout steps=480 score=83 metric=83
snake_action_scorer_config_ce_holdout steps=480 score=68 metric=68
snake_action_scorer_config_file_holdout steps=480 score=68 metric=68
```

Previous config-file Snake floor was `58`, now encoded in `tests/test_benchmark.c`.

## Lesson

For simple discrete-control projects, the action mask is a real layer, not just a guardrail. It can encode deterministic safety constraints while still letting the model choose among safe actions.

This is different from a robotics safety filter that clamps hardware commands after prediction. Here, the mask is part of model inference and training, so the learner sees the same feasible action set during both phases.
