# Tokenized Control Paper Notes

Research snapshot: 2026-07-06.

## Main takeaway

For Tokamech, copy the sequence interface, not the scale.

The useful common pattern is:

```text
fixed episodic logs
typed observation/action elements
causal history window
next action prediction
closed-loop execution with real observations
valid action constraints
```

The first line-follower system should use small discrete tokens and categorical action prediction. Continuous trajectory modeling, learned VQ tokenizers, beam-search planning, and large vision/language models are future tools, not v0 requirements.

## Humanoid Locomotion as Next Token Prediction

Sources:

- Paper: https://arxiv.org/abs/2402.19469
- Project page: https://humanoid-next-token-prediction.github.io/

Relevant idea:

The system models full sensorimotor trajectories autoregressively. Observations and actions are arranged in a sequence, and the model predicts future trajectory elements. At deployment, the controller executes the predicted action but feeds the next real observation back into the context.

Important correction for this repo:

The paper's "tokens" are not necessarily NLP-style integer vocabulary IDs. Many trajectory elements are continuous vectors projected into model space. So the big idea is not "everything must be one shared integer vocabulary." The big idea is "control can be framed as autoregressive sequence prediction."

What to copy:

- Interleave observations and actions in a trajectory sequence.
- Use short context windows.
- Support missing or irrelevant modalities with masks later.
- During deployment, execute predicted action and replace predicted observation with real measured observation.

What to avoid for v0:

- Do not start with high-dimensional continuous modality embeddings.
- Do not make next-observation prediction required for the first useful controller.
- Do not copy humanoid-scale model complexity for a four-sensor robot.

## Decision Transformer

Source:

- https://arxiv.org/abs/2106.01345

Relevant idea:

Decision Transformer treats reinforcement learning as sequence modeling. It commonly interleaves return-to-go, state, and action, then predicts the next action.

What to copy:

- Fixed sequence windows.
- Teacher-forced action prediction.
- Optional conditioning token later, such as target speed, target track quality, or mode.

What to avoid for v0:

- Do not require return-to-go conditioning before the basic imitation loop works.
- Do not make reward design a blocker for behavior cloning.

## Trajectory Transformer

Source:

- https://arxiv.org/abs/2106.02039

Relevant idea:

Trajectory Transformer flattens states, actions, and rewards into a sequence and learns next-token likelihood. It can use model predictions for planning.

What to copy:

- The "everything is a sequence" mental model.
- Discretization of continuous values can be a useful interface.

What to avoid for v0:

- Do not start with beam search or planning over imagined futures.
- Do not use one giant flat vocabulary if typed heads are easier to implement correctly.

## Behavior Transformer and VQ-style action tokenizers

Sources:

- Behavior Transformer: https://arxiv.org/abs/2206.11251
- VQ-BeT: https://arxiv.org/abs/2403.03181

Relevant idea:

For behavior cloning, continuous MSE can produce averaged actions when multiple valid behaviors exist. Discrete action bins or learned action codes can preserve multimodal behavior.

What to copy:

- Prefer categorical action prediction for simple discrete actions.
- Consider learned action codes later if actions become high-dimensional or chunked.

What to avoid for v0:

- Do not add k-means, VQ-VAE, residual decoders, or action chunks until fixed action tokens fail.

## RT-1 and RT-2

Sources:

- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818

Relevant idea:

Robot actions can be represented as discrete bins or token-like outputs, and inference must constrain output to valid robot actions.

What to copy:

- Valid action masks.
- Typed action output heads.
- Closed-loop robot execution: observe, predict action, execute, repeat.

What to avoid for v0:

- Do not add language, web data, or vision-language abstractions.
- Do not decode invalid tokens and hope the policy learns not to.

## FAST and future action-tokenizer work

Source:

- FAST: https://arxiv.org/abs/2501.09747

Relevant idea:

For high-frequency robot control, action chunks may need compression or specialized tokenization. Naive per-timestep bins can become inefficient.

What to copy later:

- If Tokamech later predicts long action chunks, consider compressed action tokens.

What to avoid for v0:

- Do not solve high-dimensional dexterous action tokenization for a line follower.

## Tokamech v0 implication

Use this first:

```text
episode -> BOS, obs/action history, next action targets, EOS
```

Recommended line-follower tokens:

```text
OBS: packed QTI sensor token
ACTION: one of 7 steering tokens
SPECIAL: PAD, BOS, EOS, MASK, RESET
```

Recommended training:

```text
input: recent observation/action tokens
target: next action token
loss: categorical cross-entropy over valid action tokens
context: 16 to 64 timesteps
```

Recommended inference:

```text
read real or simulated sensors
tokenize current observation
append to ring buffer
predict next valid action token
detokenize action token to motor command
execute command
repeat
```

