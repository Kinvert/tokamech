# Method Glossary and Learning Map

Research snapshot: 2026-07-06.

This file explains the paper and method names that show up in Tokamech planning. It is written for learning, not for claiming expert-level coverage of every paper.

Tokamech's goal is to make these ideas concrete by letting you swap one layer at a time:

```text
sequence_layout
tokenizer/vectorizer
embedder
sequence_model
prediction_head
loss
trainer
decoder
runtime
```

The right mental model is:

```text
keep most of the stack stable
replace one layer
run the same task
measure what changed
```

This is the same way MNIST can teach convolutions, CUDA matmul, image tiling, and loss functions. Tokamech should do that for sequence-model control.

## Core vocabulary

### Behavior cloning

Behavior cloning means supervised imitation learning.

Instead of learning by trial and error, the model watches examples:

```text
observation history -> expert action
```

Then it learns to predict the expert action.

For the line follower:

```text
sim oracle follows line
dataset records sensors and oracle steering actions
model learns to imitate oracle actions
```

This is the simplest starting point.

### Sequence modeling

Sequence modeling means learning from ordered data.

Examples:

```text
text: word_0, word_1, word_2
robot: observation_0, action_0, observation_1, action_1
game: frame_0, joystick_0, frame_1, joystick_1
```

The model uses past items to predict future items.

### Next-token prediction

Next-token prediction means:

```text
given previous sequence items -> predict the next item
```

In language:

```text
"the robot walks" -> predict next word
```

In control:

```text
previous observations and actions -> predict next action
```

Important: in robotics papers, "token" does not always mean a small integer ID. It can also mean a continuous vector representing one timestep or one modality.

### Transformer

A Transformer is a sequence model based on attention.

The useful intuition:

```text
each item in the sequence can look back at earlier items
attention decides which earlier items matter
```

For control:

```text
current action can depend on recent sensor readings, previous actions, and commands
```

### Attention

Attention is a learned lookup over context.

At each sequence position, the model asks:

```text
which previous positions are useful for predicting this position?
```

The implementation uses query, key, and value vectors, but the intuition is enough for now:

```text
query: what am I looking for?
key: what does this past item contain?
value: what information should I copy/use if this past item matters?
```

### KV cache

KV cache is an inference optimization for attention models.

Without cache:

```text
recompute attention for the whole history every timestep
```

With cache:

```text
store old key/value vectors
process only the newest item
```

This matters later for fast realtime control. It is not the same thing as an embedding table.

### Embedding

An embedding is the model's internal vector for an input item.

Example:

```text
OBS token id 57 -> embedding vector of size 64
```

The token ID is inspectable. The embedding vector is learned model state.

### Decoder

In Tokamech planning, decoder means:

```text
model output -> executable action or plan
```

Examples:

```text
action token -> motor PWM pair
VQ code -> continuous robot action
diffusion output -> action trajectory
beam search output -> first action in planned path
```

## Papers and method families

## Decision Transformer

Paper:

```text
https://arxiv.org/abs/2106.01345
```

Plain-language idea:

Decision Transformer treats reinforcement learning data like a sequence modeling problem.

Instead of learning a value function or policy gradient, it trains a Transformer on sequences like:

```text
return_to_go, state, action, return_to_go, state, action
```

Then it asks:

```text
if I want this much future return, and I am in this state, what action should come next?
```

Tokamech layer focus:

```text
sequence_layout
embedder
sequence_model
prediction_head
loss
trainer
```

Why it is useful for learning:

You can keep the model mostly the same and change the sequence layout from plain behavior cloning to return-conditioned behavior cloning.

Good experiment:

```text
same dataset
same model
compare:
  state -> action
  return, state, action -> next action
```

## Trajectory Transformer

Paper:

```text
https://arxiv.org/abs/2106.02039
```

Plain-language idea:

Trajectory Transformer models full trajectories:

```text
states, actions, rewards, future states, future actions
```

It can then use search/planning over predicted futures.

Instead of only asking:

```text
what is the next action?
```

it can ask:

```text
which future action sequence looks best?
```

Tokamech layer focus:

```text
sequence_layout
tokenizer/vectorizer
sequence_model
prediction_head
decoder
```

Why it is useful for learning:

It teaches that decoding can be more than argmax. The same model can be paired with different decoders:

```text
greedy next action
sampling
beam search
receding-horizon planning
```

Risk:

Planning over generated futures can be expensive and brittle.

## Behavior Transformer / BeT

Paper:

```text
https://arxiv.org/abs/2206.11251
```

Plain-language idea:

Behavior cloning with plain MSE can average actions badly.

Example:

```text
valid behavior A: go left around obstacle
valid behavior B: go right around obstacle
MSE average: go straight into obstacle
```

Behavior Transformer handles this by predicting an action mode plus a small correction.

Roughly:

```text
choose action cluster
predict residual offset
decode cluster + offset into final action
```

Tokamech layer focus:

```text
tokenizer/vectorizer
prediction_head
loss
decoder
```

Why it is useful for learning:

It is a clean example of changing action representation while leaving much of the stack alone.

Good experiment:

```text
direct MSE action regression
vs
discrete action cluster + residual
```

## VQ-BeT

Paper:

```text
https://arxiv.org/abs/2403.03181
```

Plain-language idea:

VQ-BeT is related to Behavior Transformer, but uses vector quantization to learn better action codes.

Vector quantization means:

```text
continuous action patterns -> nearest learned code
```

Instead of hand-designed bins or simple clusters, the model learns a codebook of action patterns.

Tokamech layer focus:

```text
tokenizer/vectorizer
prediction_head
loss
decoder
```

Why it is useful for learning:

It is a natural "advanced tokenizer" lesson:

```text
scalar bins -> k-means bins -> VQ codes
```

Risk:

Learned tokenizers can hide assumptions. Always measure reconstruction error before policy success.

## ACT / ALOHA

Paper:

```text
https://arxiv.org/abs/2304.13705
```

Plain-language idea:

ACT predicts a chunk of future actions instead of one action at a time.

Instead of:

```text
predict action_t
```

it predicts:

```text
predict action_t, action_t+1, ..., action_t+K
```

At runtime, overlapping chunks are combined to smooth control.

Tokamech layer focus:

```text
sequence_layout
prediction_head
loss
decoder
runtime
```

Why it is useful for learning:

It teaches compounding error, action smoothing, latency, and action chunking.

Good experiment:

```text
single-step action prediction
vs
chunked action prediction
```

## FAST

Paper:

```text
https://arxiv.org/abs/2501.09747
```

Plain-language idea:

FAST is about action tokenization for high-frequency robot actions.

Naively tokenizing every action dimension at every timestep can be inefficient. FAST compresses action sequences in frequency space, similar in spirit to representing a signal by useful frequencies instead of raw samples.

Tokamech layer focus:

```text
tokenizer/vectorizer
decoder
loss
```

Why it is useful for learning:

It is almost a perfect tokenizer lesson:

```text
raw action bins
vs
chunked bins
vs
frequency/compressed action tokens
```

Risk:

Compression can make actions harder to understand. Tokamech should include playback and reconstruction metrics.

## Mamba / Decision Mamba

Papers:

```text
https://arxiv.org/abs/2312.00752
https://arxiv.org/abs/2403.19925
```

Plain-language idea:

Mamba is a sequence model that is not attention-based. It is based on selective state-space models.

The useful comparison:

```text
Transformer: looks back over context with attention
Mamba/SSM: maintains and updates compact state over time
```

Decision Mamba applies this type of sequence model to Decision-Transformer-like reinforcement learning.

Tokamech layer focus:

```text
sequence_model
runtime state
sometimes trainer
```

Why it is useful for learning:

This is the cleanest "swap the sequence model" experiment.

Good experiment:

```text
same tokenizer
same sequence_layout
same head
same loss
compare:
  tiny Transformer
  recurrent model
  Mamba/SSM-style model
```

Runtime issue:

Transformers may use KV cache. Mamba-like models use recurrent/SSM state. The framework needs a general runtime state interface.

## RWKV

Project family:

```text
https://www.rwkv.com/
```

Plain-language idea:

RWKV is another sequence-model family that tries to get some Transformer-like benefits with recurrent-style inference.

Tokamech layer focus:

```text
sequence_model
runtime state
```

Why it is useful for learning:

It belongs in the same study bucket as Mamba:

```text
What changes when attention is replaced by streaming recurrent state?
```

## Humanoid Locomotion as Next Token Prediction

Paper:

```text
https://arxiv.org/abs/2402.19469
```

Plain-language idea:

This is the inspiration paper.

It treats humanoid locomotion as autoregressive prediction over sensorimotor trajectories:

```text
observation, action, observation, action, ...
```

In the paper, the token is closer to a continuous observation-action timestep vector than a small integer word token.

It predicts both future observations and future actions during training. At runtime, it executes the predicted action but replaces predicted observations with real robot observations.

Tokamech layer focus:

```text
sequence_layout
tokenizer/vectorizer
embedder
sequence_model
prediction_head
loss
decoder
runtime
data collection
```

Why it is useful for learning:

This is a capstone, not the first lesson. It combines many ideas:

```text
joint timestep layout
continuous vectors
modality-specific projections
missing-modality masks
next observation/action prediction
real-time robot deployment
```

Tokamech line-follower v0 is a simplified version of the same spirit, not an exact copy.

## StARformer

Paper:

```text
https://arxiv.org/abs/2110.06206
```

Plain-language idea:

StARformer builds local state-action-reward representations before doing longer-horizon sequence modeling.

It says:

```text
first understand short local groups of state/action/reward
then model the longer sequence
```

Tokamech layer focus:

```text
sequence_layout
embedder
sequence_model
```

Why it is useful for learning:

It teaches that an embedder can be more than a simple lookup or linear projection. It can include local interaction/fusion.

Risk:

The boundary between embedder and sequence model gets blurry.

## RT-1

Paper:

```text
https://arxiv.org/abs/2212.06817
```

Plain-language idea:

RT-1 is a large-scale real-robot Transformer policy. It uses robot data from many tasks and predicts robot actions from observations and task conditioning.

Tokamech layer focus:

```text
tokenizer/vectorizer
embedder
sequence_model
prediction_head
loss
trainer
data collection
runtime
```

Why it is useful for learning:

It teaches multimodal policy structure:

```text
image observations
language/task conditioning
discretized actions
multi-task training
```

Risk:

A small implementation can teach the architecture, but not reproduce the scaling result. The dataset scale is central.

## RT-2

Paper:

```text
https://arxiv.org/abs/2307.15818
```

Plain-language idea:

RT-2 uses a pretrained vision-language model and adapts it so robot actions can be emitted as token-like outputs.

This is closer to:

```text
foundation model + robot action interface
```

than:

```text
small control model with one swapped layer
```

Tokamech layer focus:

```text
almost everything
```

Why it is useful for learning:

It is a boundary case. It teaches what changes when the sequence model is no longer something you train locally from scratch.

Risk:

RT-2 is not a normal module swap. Web-scale pretraining and VLM infrastructure dominate the method.

## Diffusion Policy

Paper:

```text
https://arxiv.org/abs/2303.04137
```

Plain-language idea:

Diffusion Policy generates action trajectories by iterative denoising.

Instead of:

```text
predict action directly
```

it does something closer to:

```text
start with noisy action sequence
repeatedly denoise it conditioned on observations
execute part of the cleaned-up action sequence
```

Tokamech layer focus:

```text
prediction_head
loss
trainer
decoder
sequence_layout
runtime
```

Why it is useful for learning:

It teaches a very different way to handle multimodal continuous actions.

Risk:

It is not a simple next-token head. It changes training loss and inference loop together.

## TD-MPC

Paper:

```text
https://arxiv.org/abs/2203.04955
```

Plain-language idea:

TD-MPC learns a latent dynamics model and uses planning to choose actions.

Instead of only learning:

```text
observation -> action
```

it learns pieces like:

```text
latent state
latent dynamics
reward prediction
value prediction
planner
```

Tokamech layer focus:

```text
world_model extension
planner
loss
trainer
runtime
replay/data collection
```

Why it is useful for learning:

It teaches where supervised sequence control ends and model-based RL begins.

Risk:

It is not honest to squeeze this into one decoder or one sequence model swap.

## Dreamer / DreamerV3

Paper:

```text
https://arxiv.org/abs/2301.04104
```

Plain-language idea:

Dreamer learns a world model, then trains behavior by imagining future rollouts inside that learned model.

Major pieces:

```text
encoder
world model
reconstruction/prediction losses
actor
critic
imagined rollouts
replay buffer
online data collection
```

Tokamech layer focus:

```text
world_model extension
actor
critic
trainer
runtime
replay/data collection
```

Why it is useful for learning:

It should become a separate learning track after sequence-first supervised control.

Risk:

It cuts across the whole system. It is not a normal layer swap.

## How these methods map to Tokamech layers

Good one-layer or adjacent-layer study cases:

```text
sequence_layout:
  Decision Transformer
  Trajectory Transformer

tokenizer/vectorizer:
  BeT
  VQ-BeT
  FAST

embedder:
  StARformer
  RT-1-style modality embeddings

sequence_model:
  Transformer
  Mamba
  RWKV
  recurrent models

prediction_head:
  MSE regression head
  categorical action head
  BeT residual head
  ACT chunk head
  diffusion noise head

loss:
  MSE
  cross entropy
  residual L1
  VQ reconstruction/commitment
  diffusion denoising

decoder:
  argmax action
  sampling
  action detokenization
  beam search
  temporal ensembling
  diffusion denoising
```

Multi-layer capstones:

```text
Humanoid next-token prediction
RT-1
Diffusion Policy
```

Separate future tracks:

```text
RT-2 and VLA foundation-model adaptation
TD-MPC
Dreamer/world models
```

## Suggested learning sequence

Start here:

```text
1. Line follower direct behavior cloning.
2. Discrete action tokens and cross-entropy.
3. Continuous action regression and MSE.
4. Joint timestep next-token prediction.
5. Decision Transformer-style return conditioning.
```

Then study action representations:

```text
6. Scalar action bins.
7. BeT-style action cluster plus residual.
8. VQ action codes.
9. FAST-style action chunk/frequency tokens.
```

Then study sequence models:

```text
10. Tiny causal Transformer.
11. Recurrent model.
12. Mamba/SSM-style model.
13. KV-cache inference versus recurrent state inference.
```

Then study decoders:

```text
14. Greedy action.
15. Sampling.
16. Beam-search planning.
17. ACT action chunks and temporal ensembling.
18. Diffusion denoising decoder.
```

Then capstones:

```text
19. RT-1-style multimodal conditioning.
20. Humanoid next-token style multimodal trajectories.
21. World-model track: TD-MPC or Dreamer.
```

## Practical rule for adding a paper

When reading a new paper, ask:

```text
Which layer did they really change?
```

If the answer is one layer:

```text
implement one module
declare input/output spec
add config option
run existing stack
```

If the answer is adjacent layers:

```text
implement a compatible module family
example: tokenizer + head + loss + decoder
```

If the answer is most of the stack:

```text
make it a capstone or extension track
do not pretend it is one interchangeable module
```

This is how Tokamech stays useful for learning without becoming a vague framework that accepts everything and explains nothing.

