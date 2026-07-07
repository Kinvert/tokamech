# Action-Conditioned Scorers

Date: 2026-07-06

## Why this is the next candidate

The current best learned Snake result came from giving the MLP explicit per-action evidence:

```text
snake_mlp_action_features_holdout steps=480 score=51 metric=51
```

That result suggests a useful next layer: instead of producing all action logits from one state vector, score each candidate action with a shared model:

```text
score = f(state_features, candidate_action_features)
chosen_action = argmax valid score
```

This is not a replacement for token sequence modeling. It is a small action head/model option that can live beside the current categorical head and MLP window model.

## Paper alignment

DQN-style value methods learn action values. The original Atari DQN paper describes a neural model whose output estimates future reward values from sensory input, then selects actions from those values (`https://arxiv.org/abs/1312.5602`).

DDPG adapts Q-learning ideas to continuous control with an actor-critic setup, where the critic evaluates state-action choices and the actor learns actions that score well (`https://arxiv.org/abs/1509.02971`).

Soft Actor-Critic also centers actor updates around Q-style action-value estimates while adding entropy for exploration and stability (`https://arxiv.org/abs/1801.01290`).

Decision Transformer points in a different direction: sequence modeling over returns, states, and actions, predicting future actions autoregressively (`https://arxiv.org/abs/2106.01345`). That supports the repo's long-term sequence-model stack, but it is heavier than the next Snake improvement needs.

BeT and VQ-BeT are relevant for action tokenization and multimodal continuous behavior generation (`https://arxiv.org/abs/2206.11251`, `https://arxiv.org/abs/2403.03181`). They are good future action-tokenizer references, but Snake's discrete action space does not need them yet.

## Recommended v0 design

Add a small CPU model called something like:

```text
core/model/action_scorer
```

The model should score one candidate action at a time with shared weights:

```text
input_dim = state_feature_count + per_action_feature_count
hidden_dim = small fixed/INI-selected value
output_dim = 1 scalar score
```

A head then loops over actions:

```text
for each action:
    if action is invalid:
        score = masked_out
    else:
        score = model(state, action_features[action])
choose argmax score
```

This is different from the current `snake_mlp_action_features_holdout`, which appends all action features into one large vector and emits four logits. The proposed scorer shares the same parameters across actions, so it has a stronger inductive bias:

```text
same question for every action: how good is this move from this state?
```

## Why this should generalize across projects

`Centerline`

```text
state = current offset
candidate action = left/stay/right command and resulting offset delta
score = how close the action gets to zero
```

`Breakout`

```text
state = ball and paddle features
candidate action = left/stay/right and next paddle position
score = whether the paddle moves toward the ball intercept
```

`Snake`

```text
state = board/head/food/direction features
candidate action = valid_now, distance delta, next relative food vector
score = planner-labeled move quality
```

That makes this a better core layer candidate than more Snake-only feature hacks.

## Training options

Start with supervised imitation:

```text
positive action: planner/expert action
negative actions: other valid actions
loss: margin ranking or binary logistic
```

A minimal first version can use a simple pairwise margin loss:

```text
loss = max(0, margin - score(expert) + score(other_valid_action))
```

This is easy to test in C and does not require bootstrapped RL targets.

Later options:

```text
cross entropy over action scores
TD/Q-learning target for real RL
conservative Q-style penalties for offline data
entropy-regularized actor-critic style heads
```

## TDD plan for implementation

First test:

```text
action scorer initializes, scores two tiny candidate actions, and argmax selects the higher score.
```

Second test:

```text
one margin update raises the expert action score above an invalid/worse action score.
```

Third test:

```text
Centerline benchmark stays at metric 1 using the scorer.
```

Fourth test:

```text
Snake action-scorer holdout benchmark reaches or beats the current no-planner baseline threshold.
```

Do not wire this into every project at once. Add one clean core scorer and one benchmark path, then expand only if the benchmark result justifies it.

## Risks

This may not beat the current appended-action-feature MLP immediately. The shared scorer may be too constrained if the action features are weak.

The first version should stay benchmark-level and CPU-only. If it performs well, move it behind the INI layer factory.

## Implementation result

The first CPU implementation now exists as `core/model/action_scorer`.

Current benchmark:

```text
snake_action_scorer_holdout steps=480 score=58 metric=58
```

The implementation kept the original four-value `snake_write_action_features` path unchanged for the appended-action MLP. A new `snake_write_action_space_features` path adds reachable-space fraction for shared action scoring.

That split matters: adding reachable-space directly to the old appended-action MLP representation regressed `snake_mlp_action_features_holdout` from `51` to `38`. The useful design is not "always add more features." The useful design is "make representations interchangeable and benchmark each one."

## Recommendation

Implement action-conditioned scoring next, but keep the scope tight:

```text
core model: shared scalar scorer
head: masked argmax over candidate scores
first benchmark: Snake action scorer holdout
cross-project check: Centerline/Breakout benchmark unchanged
```

This is the most direct next step from the current evidence. It is paper-aligned, understandable in C, and likely to teach the architecture something useful even if the first score is negative.
