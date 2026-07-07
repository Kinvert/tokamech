# Action Mask Head

Date: 2026-07-06

This note tracks the first reusable action-mask head.

## Problem

Robotics and games often have actions that are invalid in a given state:

```text
Snake: move into wall/body
robotics: exceed joint limits or command unsafe actuator range
games: choose a move that is illegal in the current state
```

The policy should still produce action logits. The runtime can then block invalid actions before decoding or actuation.

## Current API

```c
int tkm_action_mask_argmax(
    const float* values,
    const uint8_t* valid_mask,
    uint32_t count,
    uint16_t* out_index
);
```

The mask uses `1` for valid and `0` for invalid. If all actions are invalid, the function returns `TKM_ERR`.

## Snake integration

Snake now provides a project-specific mask:

```c
int snake_write_action_mask(
    const SnakeEnv* env,
    uint8_t* out_mask,
    uint32_t max_actions
);
```

It simulates each action one step and marks actions that would immediately terminal as invalid.

## Benchmark result

```text
snake_mlp_ce_masked_holdout steps=480 score=30 metric=30
```

This is weaker than planner-fallback safety:

```text
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
```

Masked cross-entropy training was also tested:

```text
snake_mlp_masked_ce_holdout steps=439 score=17 metric=17
```

For the current tiny Snake MLP, ordinary CE training plus masked evaluation is stronger than masked CE training plus masked evaluation.

DAgger-style online planner labeling improves the no-planner masked policy further:

```text
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
```

That gap is useful. It shows the planner fallback was doing real corrective work. The masked benchmark is a cleaner learned-policy test because the planner does not supply replacement actions during evaluation.

## Architectural role

Action masks belong after model logits and before the action head:

```text
features/tokens -> model -> logits -> action mask -> argmax/sample -> decoder -> env
```

This keeps safety constraints modular. The model can be swapped without rewriting the environment, and the environment can expose validity rules without knowing how the model was trained.
