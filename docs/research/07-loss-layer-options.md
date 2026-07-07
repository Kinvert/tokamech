# Loss Layer Options

Date: 2026-07-06

## Current implementation status

The first generic loss module is implemented in core:

```text
core/loss/loss.h
core/loss/loss.c
tests/test_loss.c
```

Implemented functions:

```text
tkm_loss_cross_entropy
tkm_loss_mse
tkm_loss_huber
tkm_loss_binary_cross_entropy
tkm_loss_weighted_sum
tkm_loss_action_smoothness
```

This is core math only. The trainer and ini-driven selection are not implemented yet.

## Why these losses first

These losses cover the phase-0 projects and the next likely learned models without forcing a neural-net trainer into the repo too early.

```text
cross_entropy              discrete next-token or action prediction
multi_cross_entropy        planned wrapper over multiple cross-entropy heads
mse                        continuous prediction baseline
huber                      robust continuous prediction for robotics/control
binary_cross_entropy       terminals, collisions, success flags
weighted_sum               combine action, value, terminal, smoothness, or auxiliary losses
action_smoothness          penalize sudden changes in continuous action vectors
```

`huber` is especially useful for sim2real/control work because it behaves like MSE for small errors but is less dominated by large outliers.

## Proposed ini shape

Simple single-term loss:

```ini
[loss]
kind = cross_entropy
target = next_action
```

Continuous action regression:

```ini
[loss]
kind = huber
target = action_vector
delta = 1.0
```

Terminal prediction:

```ini
[loss]
kind = binary_cross_entropy
target = terminal
```

Combined loss:

```ini
[loss]
kind = weighted_sum

[loss.term0]
name = action_ce
kind = cross_entropy
weight = 1.0

[loss.term1]
name = terminal_bce
kind = binary_cross_entropy
weight = 0.2

[loss.term2]
name = action_huber
kind = huber
weight = 0.1
delta = 1.0
```

Action smoothness as a standalone loss:

```ini
[loss]
kind = action_smoothness
action_dim = 2
```

## Current tests

`tests/test_loss.c` covers:

```text
cross entropy reads the target probability
MSE averages squared error
Huber is quadratic near zero and linear for large errors
Binary cross entropy handles terminal-style targets
Weighted sum combines separate terms
Action smoothness penalizes action deltas over time
Bad inputs return TKM_ERR
```

## What is intentionally not implemented yet

These are useful later, but not needed for the first core loss module:

```text
softmax_cross_entropy_from_logits
multi_cross_entropy helper
gaussian_negative_log_likelihood
mixture_density_negative_log_likelihood
vq_codebook_loss
vq_commitment_loss
contrastive_infonce
ppo_clipped_policy_loss
value_loss
entropy_bonus
kl_penalty
energy_penalty
jerk_penalty
constraint_violation_loss
```

## Architecture rule

Losses should stay independent from projects.

Projects define what targets mean:

```text
centerline: action class, offset regression, terminal flag
breakout: action class, brick/contact auxiliary targets
snake: action class, terminal flag, food/reward auxiliary targets
line_follower: motor command tokens, QTI predictions, smoothness terms
```

Core loss code should only compute math over arrays and scalars. It should not know what a QTI sensor, paddle, snake body, or motor is.
