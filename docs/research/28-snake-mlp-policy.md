# Snake MLP Policy Benchmark

Date: 2026-07-06

This note tracks the first Snake policy benchmark that uses the CPU MLP training path instead of exact-token lookup or nearest-neighbor imitation.

## Layer stack

```text
Snake raw state
-> snake_write_features
-> one-hidden-layer ReLU MLP
-> categorical argmax action head
-> immediate safety filter
-> snake_step_env
```

The model is trained by planner imitation:

```text
planner rollout state features -> planner action
```

The benchmark excludes the five evaluation first-food starts from MLP training:

```text
{6,2}, {3,8}, {8,8}, {1,1}, {1,8}
```

Training uses the other interior food starts that do not overlap the fixed starting snake body.

## What this proves

This proves the core can run a continuous-feature neural policy end to end on Snake:

```text
features -> trainable model -> logits -> action -> env
```

It is a stronger learned-policy test than exact full-state token lookup because the MLP must map continuous features to actions instead of memorizing exact hashed states.

Current result:

```text
snake_mlp_safe_holdout steps=480 score=46 metric=46
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
snake_mlp_ce_masked_holdout steps=480 score=30 metric=30
```

The current nearest-neighbor plus safety benchmark scores `47` on the same aggregate held-out horizon. The MSE-trained MLP is close to that continuous-memory baseline, and the cross-entropy-trained MLP slightly exceeds it while using the same compact trainable model shape.

The masked benchmark is lower, but it removes planner fallback from evaluation. The model still cannot choose immediately terminal actions, but when its first choice is invalid the core action-mask head picks the next-best valid model logit instead of asking the planner what to do.

## What this does not prove

This is still not a final Snake learner.

The policy is trained from a hand-coded planner. The safety filter still blocks immediately terminal actions. The benchmark measures whether this layer stack can imitate useful behavior on held-out starts, not whether it discovers the behavior from reward.

## Why keep it

This benchmark is a useful bridge for later papers and layer experiments:

```text
replace MLP with transformer / Mamba / RNN
replace MSE imitation with cross entropy or behavior-token loss
replace hand features with VQ or patch tokens
keep Snake benchmark and safety/runtime loop stable
```
