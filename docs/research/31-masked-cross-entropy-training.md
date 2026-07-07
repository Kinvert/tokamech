# Masked Cross-Entropy Training

Date: 2026-07-06

This note tracks masked categorical cross-entropy training for the CPU MLP layer.

## What changed

The MLP now has a training function that accepts a valid-action mask:

```c
int tkm_mlp_window_train_masked_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    const uint8_t* valid_mask,
    uint16_t target_index,
    float learning_rate
);
```

It computes softmax only over valid actions:

```text
valid action: contributes to softmax and gradient
invalid action: zero probability mass, zero gradient
invalid target: training error
empty mask: training error
```

## Cross-project benchmark results

```text
centerline_mlp_masked_ce_trained steps=8 score=1 metric=1
breakout_mlp_masked_ce_trained steps=220 score=2 metric=3
snake_mlp_masked_ce_holdout steps=439 score=17 metric=17
```

Centerline and Breakout use all-valid masks, so masked CE behaves like ordinary CE there.

Snake uses real immediate-action masks during training and masked argmax during evaluation.

## Negative Snake result

Masked CE training is currently worse than unmasked CE training with masked evaluation:

```text
snake_mlp_ce_masked_holdout steps=480 score=30 metric=30
snake_mlp_masked_ce_holdout steps=439 score=17 metric=17
```

This is useful evidence. It means the method is implemented, but it is not the better choice for the current tiny Snake MLP.

Likely reasons:

```text
the planner labels already avoid most invalid actions
the tiny MLP benefits from seeing all logits compete during ordinary CE
masked training changes the imitation data distribution when planner labels conflict with immediate masks
```

## Architectural role

Masked CE should stay available as an interchangeable training option.

It is likely useful later for environments with many illegal actions, variable action sets, or robotics constraints. It should not replace ordinary CE as the current default for Snake.

The stronger path after this result is DAgger-style online labeling:

```text
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
```
