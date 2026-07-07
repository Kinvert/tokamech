# MLP Cross-Entropy Training

Date: 2026-07-06

This note tracks the first categorical cross-entropy training path for the CPU MLP layer.

## Why this matters

The earlier MLP trainer used mean-squared error against one-hot action targets. That can work for tiny imitation tasks, but action prediction is a categorical classification problem:

```text
features -> logits -> categorical action
```

Cross entropy is the standard loss for this shape. Its logit gradient is simple:

```text
softmax(logits) - one_hot(target_action)
```

That makes it a useful phase-0 policy-imitation option before adding larger sequence models.

## Current API

```c
int tkm_mlp_window_train_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    uint16_t target_index,
    float learning_rate
);
```

The function is deterministic, CPU-only, and uses a numerically stable softmax by subtracting the maximum logit before exponentiation.

## Benchmark results

```text
centerline_mlp_ce_trained steps=8 score=1 metric=1
breakout_mlp_ce_trained steps=220 score=2 metric=3
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
```

The Snake result is the best current learned-policy benchmark in the repo:

```text
snake_nearest_safe_holdout steps=480 score=47 metric=47
snake_mlp_safe_holdout steps=480 score=46 metric=46
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
```

Masked cross-entropy training was added later as a separate option. It is useful architecturally, but it is not the current best Snake trainer:

```text
snake_mlp_masked_ce_holdout steps=439 score=17 metric=17
```

## Interpretation

Cross entropy should be the default supervised action-imitation loss for categorical actions.

Mean-squared error remains useful as a teaching baseline because it is easy to inspect and debug, but it is no longer the best default for action logits.

## Limits

This is still behavior cloning from planner actions. It does not learn from reward directly, and the Snake benchmark still uses the immediate safety filter during evaluation.
