# Tokamech Architecture Spec

Status: draft

Date: 2026-07-06

## Purpose

Tokamech is a C/raw-CUDA-first research framework for building end-to-end control systems from environments, trajectory data, token or vector representations, sequence models, training loops, and deployable policies.

The first project is a tokenized sim2real line follower. The larger goal is not a line-follower-only codebase. The larger goal is a clean modular system where different research methods can be swapped through configuration while sharing the same env/data/runtime foundation.

## Core idea

Control is treated as sequence modeling over sensorimotor experience.

An environment produces raw frames:

```text
observation_t, action_t, reward_t, terminal_t, log_t
```

A representation pipeline converts those flat frames into model-ready sequence items:

```text
env/project writes RawFrame flat buffers
-> sequence_layout selects/orders fields across timesteps
-> tokenizer/vectorizer transforms selected fields/items
-> embedder maps representations to hidden vectors
-> sequence model
-> head
-> decoder
-> action suggestion
-> optional safety filter
-> final action
```

Different methods may represent the sequence differently:

```text
separate sequence_layout:
  OBS_t, ACTION_t, OBS_t+1, ACTION_t+1

joint timestep sequence_layout:
  STEP_t = [OBS_t, ACTION_t], STEP_t+1 = [OBS_t+1, ACTION_t+1]

flat scalar stream:
  obs_dim0_t, obs_dim1_t, action_dim0_t, reward_t, ...

state-action-reward sequence_layout:
  RETURN_t, STATE_t, ACTION_t, REWARD_t
```

The core must support these layouts without hardcoding one paper's assumptions.

The sequence layout operates on raw frame fields and timestep structure. The tokenizer/vectorizer then decides how the selected fields become discrete IDs, continuous vectors, learned codes, or joint timestep representations.

## Design principles

### Interchangeable, not shapeless

Modules should be interchangeable only when their declared input and output specs match.

Research survey:

```text
docs/research/04-modular-paper-architecture-survey.md
```

That survey categorizes which papers fit this layer-by-layer architecture, which papers require coordinated module changes, and which papers are closer to full paradigm shifts.

Bad:

```text
any module can connect to any module and we hope it works
```

Good:

```text
each module declares what it consumes and produces
startup validates the full pipeline
invalid configs fail before training starts
```

Do not force every layer through one generic `forward()` interface. Use a shared spec/registry system, but typed interfaces for each layer:

```text
TkmTokenizer
TkmEmbedder
TkmModel
TkmHead
TkmLoss
TkmDecoder
TkmTrainer
```

These interfaces may share common fields such as `name`, `describe`, `init`, and `close`, but their core functions should match the job of the layer.

### Presets first, overrides second

Users should be able to start with a known-good method:

```ini
[method]
preset = line_follower_discrete_bc
```

Then override one layer:

```ini
[method]
preset = line_follower_discrete_bc

[model]
name = mingru
```

Advanced users can specify the whole pipeline manually:

```ini
[sequence_layout]
name = joint_timestep

[tokenizer]
name = qti_packed_discrete

[embedder]
name = learned_table

[model]
name = tiny_causal_transformer

[head]
name = discrete_action_head

[loss]
name = action_cross_entropy

[trainer]
name = offline_bc
```

### Project-specific meaning, shared mechanics

The core should not know what a QTI sensor, humanoid joint, Atari frame, or joystick button means.

Projects own domain meaning:

```text
line_follower owns QTI calibration and steering actions
atari owns frame/action preprocessing
humanoid owns proprioception/action vector schemas
```

Projects should expose that meaning through simple flat buffers, not through hidden framework magic.

PufferLib-style project code should be obvious:

```c
void line_follower_observe(LineFollower* env, TkmObs* obs) {
    obs->u16[0] = env->qti[0];
    obs->u16[1] = env->qti[1];
    obs->u16[2] = env->qti[2];
    obs->u16[3] = env->qti[3];
}
```

Those values are raw observation channels, not model tokens yet. The configured tokenizer/vectorizer turns the flat observation buffer into tokens or vectors.

The core owns mechanics:

```text
buffers
sequence windows
batching
config validation
training loop dispatch
reusable tokenizer/vectorizer algorithms
model execution
logging
profiling
checkpoint IO
```

Core mechanics must stay reusable. Do not put Snake, line-follower, Breakout, QTI, Atari, humanoid, or any other project-specific assumptions into `core/`.

If a method sounds project-shaped but is actually general, name and implement the general version. For example, DAgger-style policy-induced-state training should not be `snake_dagger`. It should be a reusable trainer pattern that can ask any project for:

```text
reset/start states
observe/features
valid-action mask
expert/planner label
step(action)
```

Snake can provide those hooks from `projects/snake/`, but the reusable loop belongs in shared trainer/runtime code only after its interface is project-agnostic.

### C first, raw CUDA where it matters

The hot path should use:

```text
plain C structs
raw pointers
fixed-size buffers
explicit ownership
minimal hidden allocation
deterministic memory layout
CUDA kernels for training/model hot paths
```

Python may exist later for convenience, plotting, or orchestration, but it should not define the core architecture.

The first vertical slice may use CPU reference implementations. Core APIs should still be written so CUDA backends can replace specific layers later without changing project code or configs.

Performance work should be evidence-driven:

```text
start with deterministic CPU reference
add CUDA for one layer at a time
record before/after numbers in benchmark notes
keep reference tests for correctness
```

Determinism matters:

```text
fixed seeds
stable dataset ordering
deterministic test envs
explicit RNG state
repeatable CPU reference paths
```

### Test-driven implementation

Tokamech should be built with TDD for production code.

Rule:

```text
no production code without a failing test first
```

Development loop:

```text
1. RED: write a small test for the next behavior.
2. Verify the test fails for the expected reason.
3. GREEN: write the minimum code to pass.
4. Verify the test passes.
5. Refactor only while tests stay green.
```

This applies to:

```text
core modules
env APIs
tokenizers/vectorizers
sequence layouts
dataset IO
trainers
decoders
runtime behavior
bug fixes
refactors
```

Exceptions require explicit agreement before implementation:

```text
throwaway prototypes
generated code
documentation-only changes
configuration-only changes
```

Phase 0 exists partly to make TDD cheap. The `centerline` project should be small enough that end-to-end behavior can run as a deterministic test.

## Terms

### Raw frame

The direct environment result at one timestep.

Example:

```text
qti_raw = [812, 644, 210, 188]
action = slight_left
reward = 0.81
terminal = false
logs = optional debug or aggregate fields
```

Project code should write raw observations and actions into explicit flat buffers. For example, a line follower can expose four QTI channels as:

```text
obs.u16[0] = qti_left_outer
obs.u16[1] = qti_left_inner
obs.u16[2] = qti_right_inner
obs.u16[3] = qti_right_outer
```

Changing the tokenizer in config should not require rewriting this observation function.

### Sequence item

A model-facing item produced from raw frames. A sequence item can be a discrete token ID, a continuous vector, a learned code, or a structured timestep.

Examples:

```text
OBS(57)
ACTION(3)
STEP([qti0, qti1, qti2, qti3, action])
REWARD(0.81)
MASK
```

### Token

In Tokamech, "token" is a broad term. It does not always mean a language-model integer ID.

Allowed token-like representations:

```text
DISCRETE_ID        integer vocabulary ID
CONTINUOUS_VECTOR  fixed-size numeric vector
VQ_CODE            learned vector-quantized code ID
JOINT_STEP         structured timestep containing multiple modalities
SPECIAL            BOS, EOS, PAD, MASK, RESET
```

### BOS and EOS

`BOS` means beginning of sequence.

`EOS` means end of sequence.

They mark boundaries in episode or training-window data:

```text
BOS, STEP_0, STEP_1, STEP_2, EOS
```

## Pipeline layers

### 1. Env

Owns simulation or hardware interaction.

Contract:

```text
reset(seed) -> first raw observation
step(action) -> next raw observation, reward, terminal, logs
close()
```

Env requirements:

```text
deterministic under fixed seed when possible
no model-specific logic
explicit flat observation/action buffers
flat data access for high throughput
standalone executable support for debugging
```

### 2. Data collector

Runs envs and writes trajectory logs.

Sources may include:

```text
oracle controller
PD controller
human input
learned policy
real robot logs
offline datasets
```

For v0, the line follower should collect data from a simulated oracle/PD controller.

### 3. Sequence layout

Defines how raw frame fields are arranged into sequence items.

Example sequence layouts:

```text
separate_obs_action
joint_timestep
flat_scalar_stream
state_action_reward
decision_transformer
```

The sequence layout does not decide how values are embedded. It decides sequence structure.

### 4. Tokenizer or vectorizer

Converts flat raw buffers into the representation expected by the model.

This layer should mostly be reusable core code, not project-specific code.

Tokamech uses `tokenizer/vectorizer` as one layer name because different papers use different words. In code, the interface can be named `tokenizer` as long as the docs state that tokenizers may produce either discrete tokens or continuous vectors.

Examples:

```text
uniform_bins
packed_bins
continuous_vector
uniform_scalar_bins
quantile_scalar_bins
one_hot
image_patches
vq_action_codes
```

A tokenizer/vectorizer declares output specs:

```text
representation kind
shape
vocab size if discrete
dtype
valid range
modality role
```

Example line-follower observation buffer:

```text
raw obs = [812, 644, 210, 188]
```

Different config-selected tokenizers can consume the same raw buffer:

```text
uniform_bins       -> [3, 2, 0, 0]
packed_bins        -> OBS(11)
continuous_vector  -> [0.812, 0.644, 0.210, 0.188]
```

The env code stays the same. The tokenizer/vectorizer changes by config.

### 5. Embedder

Maps representation items into model hidden vectors.

Examples:

```text
learned embedding table for DISCRETE_ID
linear projection for CONTINUOUS_VECTOR
modality-specific projection for JOINT_STEP
sinusoidal or fixed features
custom CUDA embedder
```

### 6. Sequence model

Consumes embedded sequence windows and produces hidden states.

Examples:

```text
lookup table
MLP over short history
causal transformer
MinGRU
RWKV-like model
custom recurrent model
```

The model should not know line-follower semantics.

### 7. Prediction head

Converts hidden states into predicted outputs.

Examples:

```text
discrete action logits
joint timestep regression
multi-modal heads
VQ code logits plus residual
reward/value head
```

### 8. Loss

Computes training objective from predictions and targets.

Examples:

```text
action cross entropy
joint timestep MSE
masked modality loss
cross entropy plus residual L1
return-conditioned behavior cloning loss
```

### 9. Trainer

Runs optimization over datasets.

Examples:

```text
offline behavior cloning
joint next-token prediction
decision transformer style training
future RL fine-tuning
```

### 10. Decoder

Converts model output into an environment or robot action.

Examples:

```text
steering token -> motor PWM pair
continuous vector -> motor command
VQ code -> action chunk
action logits -> sampled valid action
```

### 11. Safety filter

Optional project-specific final gate before an action reaches the simulator or hardware.

Input:

```text
decoded action suggestion
```

Output:

```text
final action command
```

The safety filter may pass, clamp, replace, or stop an action.

For robotics, the learned policy should not write directly to motors. Later robotics projects can expand this layer with arming, kill switch, PWM limits, line-lost stop, watchdog timeout, and action rate limits.

Safety filtering is not a required Phase 0 feature for the `centerline` test project.

## Module specs

Every interchangeable module must publish a machine-checkable spec.

Minimum spec fields:

```text
name
module_type
accepted_input_kinds
produced_output_kind
shape
dtype
vocab_size if applicable
loss_compatibility
trainable_parameters yes/no
runtime_supported yes/no
cuda_supported yes/no
```

Validation must check at least:

```text
dtype compatibility
shape compatibility
vocab size compatibility for discrete IDs
batch and time dimension expectations
required raw frame fields
required masks
action decode availability
loss and head compatibility
trainer and loss compatibility
CPU/CUDA backend availability
train/runtime support
streaming state requirements
determinism requirements for tests
```

Example:

```text
module: packed_bins
type: tokenizer
input:
  dtype: uint16
  size: 4
output:
  kind: DISCRETE_ID
  role: OBS
  dtype: uint16
  vocab_size: 256
```

Example:

```text
module: learned_table
type: embedder
accepts:
  kind: DISCRETE_ID
output:
  kind: CONTINUOUS_VECTOR
  dtype: float32
  size: hidden_size
```

Example invalid config:

```text
tokenizer output: CONTINUOUS_VECTOR
embedder: learned_table
```

Startup error:

```text
Config error:
  learned_table accepts DISCRETE_ID
  previous module produced CONTINUOUS_VECTOR
```

## C module interface sketch

The exact API will be designed during implementation. Do not force every layer into one universal function pointer shape. The shared parts should look close to:

```c
typedef struct {
    const char* name;
    const char* module_type;
    TkmSpec (*describe)(const TkmSpec* input, const TkmConfig* cfg);
    int (*init)(void* module, const TkmSpec* input, const TkmConfig* cfg);
    void (*close)(void* module);
} TkmModule;
```

Each layer then gets a typed interface. For example:

```c
typedef struct {
    TkmModule base;
    int (*encode)(void* module, const TkmRawBatch* input, TkmRepBatch* output);
} TkmTokenizer;

typedef struct {
    TkmModule base;
    int (*forward)(void* module, const TkmHiddenBatch* input, TkmHiddenBatch* output);
    int (*stream)(void* module, TkmRuntimeState* state, const TkmHiddenBatch* input, TkmHiddenBatch* output);
} TkmModel;

typedef struct {
    TkmModule base;
    int (*compute)(void* module, const TkmPredBatch* pred, const TkmTargetBatch* target, TkmLossResult* result);
} TkmLoss;
```

Registration should start static and explicit:

```c
void tkm_register_builtin_modules(TkmRegistry* registry);
int tkm_register_module(TkmRegistry* registry, const TkmModule* module);
```

Dynamic shared-object plugins can come later.

## Configuration model

Use INI for human-editable experiment configs.

Configuration levels:

```text
Level 1: preset
Level 2: preset plus overrides
Level 3: explicit pipeline
```

Example v0 preset:

```ini
[method]
preset = line_follower_discrete_bc

[run]
seed = 1
```

Equivalent explicit pipeline:

```ini
[env]
name = line_follower

[data]
source = sim_oracle
format = trajectory_v1

[sequence_layout]
name = joint_timestep_discrete
context_steps = 32

[tokenizer]
name = packed_bins
input_dtype = u16
input_size = 4
bits_per_sensor = 2

[action]
name = discrete_steering
num_actions = 8

[embedder]
name = learned_table
hidden_size = 64

[model]
name = tiny_causal_transformer
layers = 2
heads = 2
hidden_size = 64

[head]
name = discrete_action_head

[loss]
name = action_cross_entropy

[trainer]
name = offline_bc

[decoder]
name = steering_token_to_pwm
```

Paper-faithful continuous joint-timestep variant:

```ini
[sequence_layout]
name = joint_timestep

[tokenizer]
name = continuous_vector

[embedder]
name = linear_projection

[model]
name = tiny_causal_transformer

[head]
name = joint_step_regression_head

[loss]
name = mse_joint_step
```

Hybrid variant:

```ini
[sequence_layout]
name = separate_obs_action

[tokenizer.observation]
name = continuous_vector

[tokenizer.action]
name = discrete_steering

[embedder.observation]
name = linear_projection

[embedder.action]
name = learned_table

[head]
name = discrete_action_head

[loss]
name = action_cross_entropy
```

## Phase 0 project: centerline

Before the line follower, Tokamech should include an even smaller deterministic project for the TDD loop.

Goal:

```text
prove env -> data -> tokenizer -> trainer -> decoder -> runtime without physics or robot details
```

Concept:

```text
state: signed integer offset from centerline
observation: one i16 value, offset
actions: left, stay, right
transition: action moves offset toward or away from zero
oracle: if offset < 0 choose right, if offset > 0 choose left, else stay
terminal: fixed horizon or offset outside bounds
```

Why this exists:

```text
fast deterministic test fixture
small enough to inspect every buffer by hand
works with packed/discrete and continuous tokenizers
can use a lookup-table trainer before neural training exists
keeps line-follower complexity out of core TDD
```

Proposed files:

```text
projects/
  centerline/
    centerline.c
    centerline.h
    centerline.ini
    README.md
    data/
      README.md
```

Generated data for this project should live under `projects/centerline/data/` or another ignored project-local path.

## First robotics project: line_follower

### Goal

Prove the end-to-end loop:

```text
sim env -> oracle data -> sequence dataset -> train model -> closed-loop sim policy
```

### V0 data source

Use a PufferLib-style line-follower simulation environment to generate trajectories.

Data generator:

```text
hand-coded oracle or PD controller
discrete steering actions
raw QTI sensor logs
tokenized sequence logs
fixed seeds for deterministic replay
```

### V0 representation

Recommended first method:

```text
sequence_layout: joint_timestep_discrete
observation buffer: four raw QTI channels
tokenizer: reusable packed_bins tokenizer
action: discrete steering token
target: next action
loss: cross entropy
```

This is not an exact copy of Humanoid Locomotion as Next Token Prediction. It is a simpler discrete version that proves the same system shape.

### Alternative method to keep in design

Paper-faithful method:

```text
sequence_layout: joint_timestep
observation: continuous vector
action: continuous or normalized vector
target: next joint timestep
loss: MSE
```

This should be supported by the architecture, but it does not need to be implemented before the first hello-world works.

## Capstone compatibility: Humanoid Locomotion as Next Token Prediction

The architecture must support the core shape of Humanoid Locomotion as Next Token Prediction:

```text
https://arxiv.org/abs/2402.19469
```

This paper should be treated as a capstone multi-layer method, not as a single module swap.

### Paper shape

The paper frames control as autoregressive prediction over sensorimotor trajectories:

```text
(o_1, a_1, o_2, a_2, ..., o_T, a_T)
```

One paper-faithful Tokamech representation is:

```text
STEP_t = [observation_t, action_t]
STEP_t+1 = [observation_t+1, action_t+1]
```

The model predicts the next timestep token:

```text
history of STEP tokens -> predicted STEP_t+1
```

Unlike the discrete line-follower v0, this method primarily uses continuous vectors and MSE-style regression.

### Tokamech layer mapping

Required module family:

```text
sequence_layout:
  joint_timestep or multimodal_joint_timestep

tokenizer/vectorizer:
  continuous observation/action vectors
  normalized modality fields
  optional masks for missing modalities

embedder:
  linear projection
  modality-specific projection
  timestep/type embeddings

sequence_model:
  causal transformer first
  later compatible with recurrent or state-space backbones

prediction_head:
  joint timestep regression head
  or separate modality heads for observation and action

loss:
  MSE over predicted continuous fields
  masked MSE for missing or ignored modalities
  optional action-weighted loss if action accuracy matters more than observation prediction

decoder:
  extract predicted action fields
  convert predicted action vector to executable controls

runtime:
  execute predicted action
  discard predicted observation
  read real next observation from env or robot
  append real observation plus executed action back into context
```

### Runtime rule

At deployment, predicted observations are not trusted as sensor truth.

The closed-loop runtime should behave like:

```text
1. read real observation_t
2. append observation_t and previous action to context
3. model predicts observation_t+1 and action_t+1
4. decoder extracts action_t+1
5. safety filter checks action_t+1
6. env or robot executes action_t+1
7. read real observation_t+1
8. append real observation_t+1, not predicted observation_t+1
```

This rule must be explicit because it separates training-time trajectory prediction from runtime control.

### Why this matters for v0

The line-follower v0 may start simpler:

```text
packed discrete observation token
discrete steering action token
predict next action only
cross entropy loss
```

But the core cannot assume:

```text
all tokens are integer IDs
all losses are cross entropy
all heads predict only actions
all layouts keep observations and actions separate
```

The architecture must keep room for:

```text
continuous vectors
joint timestep tokens
masked modalities
next observation prediction
modality-specific heads
closed-loop real-observation override
```

## Initial repo layout

Proposed v0 layout:

```text
core/
  common/
    types.h
    status.h
    array.h
    rng.h

  config/
    config.h
    config.c
    ini.h
    ini.c
    preset.h
    preset.c

  module/
    spec.h
    module.h
    registry.h
    registry.c
    validate.h
    validate.c
    pipeline.h
    pipeline.c

  env/
    env.h
    spaces.h
    buffers.h
    vecenv.h
    vecenv.c
    logs.h

  sequence/
    sequence.h
    layout.h
    layout.c
    window.h
    window.c
    mask.h

  dataset/
    trajectory.h
    trajectory.c
    dataset.h
    dataset.c
    batch.h
    batch.c

  tokenizer/
    tokenizer.h
    uniform_bins.c
    packed_bins.c
    continuous_vector.c
    one_hot.c

  embedder/
    embedder.h
    learned_table.c
    linear_projection.c

  model/
    model.h
    mlp.c
    transformer.c
    recurrent.c

  head/
    head.h
    discrete_action_head.c
    regression_head.c
    joint_step_head.c

  loss/
    loss.h
    cross_entropy.c
    mse.c
    masked_mse.c

  decoder/
    decoder.h
    argmax.c
    sample.c
    lookup_action.c

  train/
    trainer.h
    offline_bc.c
    optimizer.h
    optimizer.c
    checkpoint.h
    checkpoint.c

  runtime/
    policy.h
    policy.c
    runtime_state.h
    safety.h
    loop.h
    loop.c

  cuda/
    cuda_common.h
    kernels.h
    matmul.cu
    softmax.cu
    attention.cu

projects/
  centerline/
    centerline.c
    centerline.h
    centerline.ini
    README.md
    data/
      README.md

  line_follower/
    line_follower.c
    line_follower.h
    line_follower.ini
    README.md
    data/
      README.md
    docs/

docs/
  research/
  design/

apps/
  collect/
  train/
  replay/
  profile/

tools/
  scripts/

tests/
  core/
  projects/
```

If a project grows several experiment configs, it can add a `configs/` directory later. V0 should keep the first `.ini` directly beside the project source.

Project-generated data should be project-local by default:

```text
projects/<name>/data/
```

Large generated files, checkpoints, and run artifacts should be ignored by git unless intentionally promoted to tiny fixtures.

`apps/` contains compiled entrypoints:

```text
apps/collect:
  run env plus oracle or policy and write trajectories

apps/train:
  load data, build configured pipeline, train, write checkpoint

apps/replay:
  replay trajectories or checkpoints for deterministic debugging

apps/profile:
  measure env steps/sec, tokens/sec, latency, and backend timings
```

`tools/` is only for helper scripts around those apps.

### `core/common/`

Small shared types and utilities.

```text
types.h:
  fixed-width aliases, dtype enums, shape structs, constants

status.h:
  error/status codes and helper macros

array.h:
  lightweight typed flat array views

rng.h:
  deterministic RNG interface for envs and sampling
```

### `core/config/`

Human-readable configuration loading and preset expansion.

```text
config.h / config.c:
  typed config structs used by the runtime

ini.h / ini.c:
  minimal INI parser or wrapper

preset.h / preset.c:
  maps preset names to full pipeline defaults
```

### `core/module/`

The compatibility layer that makes methods interchangeable without becoming shapeless.

```text
spec.h:
  machine-checkable input/output specs
  examples: DISCRETE_ID, CONTINUOUS_VECTOR, vocab_size, shape, dtype

module.h:
  shared module metadata and lifecycle hooks
  typed layer interfaces live in their own layer headers

registry.h / registry.c:
  name-to-module registry
  examples: "packed_bins", "linear_projection", "tiny_transformer"

validate.h / validate.c:
  checks configured modules connect correctly before training starts

pipeline.h / pipeline.c:
  builds the configured stack from INI and module registry
```

### `core/env/`

Generic environment ABI and vectorized environment runtime.

This directory does not contain concrete environments.

```text
env.h:
  reset, step, close interface

spaces.h:
  observation/action metadata such as dtype, size, bounds

buffers.h:
  flat buffers for observations, actions, rewards, terminals

vecenv.h / vecenv.c:
  runs many env instances against shared buffers

logs.h:
  generic metrics/log side channel
```

### `core/sequence/`

Turns episodes into model-facing sequence structure.

```text
sequence.h:
  generic sequence item and sequence batch types

layout.h / layout.c:
  sequence layouts such as separate_obs_action, joint_timestep, decision_transformer

window.h / window.c:
  fixed-context window slicing for training

mask.h:
  padding, causal, modality, and valid-token masks
```

### `core/dataset/`

Source code for trajectory storage, loading, and batching. Actual datasets should live outside core, usually in ignored `datasets/` or `runs/` paths.

```text
trajectory.h / trajectory.c:
  episode/trajectory file format and IO

dataset.h / dataset.c:
  dataset indexing, train/validation split, metadata

batch.h / batch.c:
  builds training batches from trajectory windows
```

### `core/tokenizer/`

Reusable tokenizers and vectorizers that consume flat buffers.

```text
tokenizer.h:
  tokenizer/vectorizer interface

uniform_bins.c:
  maps scalar values to fixed bins

packed_bins.c:
  packs several binned dimensions into one discrete token ID

continuous_vector.c:
  normalizes flat numeric buffers into continuous vectors

one_hot.c:
  expands discrete IDs into one-hot vectors when useful
```

### `core/embedder/`

Maps tokenizer/vectorizer outputs into model hidden vectors.

```text
embedder.h:
  embedder interface

learned_table.c:
  embedding table for discrete IDs

linear_projection.c:
  learned projection for continuous vectors
```

### `core/model/`

Sequence backbones.

```text
model.h:
  common model interface

mlp.c:
  simple history MLP baseline

transformer.c:
  causal transformer baseline

recurrent.c:
  simple recurrent baseline
```

### `core/head/`

Converts model hidden states into predictions.

```text
head.h:
  prediction head interface

discrete_action_head.c:
  logits over discrete action tokens

regression_head.c:
  continuous vector prediction

joint_step_head.c:
  predicts observation/action timestep vectors
```

### `core/loss/`

Training objectives.

```text
loss.h:
  loss interface

cross_entropy.c:
  categorical token/action loss

mse.c:
  continuous vector regression loss

masked_mse.c:
  MSE with modality or padding masks
```

### `core/decoder/`

Turns predictions into selected actions or plans.

```text
decoder.h:
  decoder interface

argmax.c:
  chooses highest-logit action

sample.c:
  samples from logits with temperature

lookup_action.c:
  maps discrete action IDs to configured action values
```

### `core/train/`

Training loop, optimizer, and checkpoint code.

```text
trainer.h:
  trainer interface

offline_bc.c:
  offline behavior cloning trainer

optimizer.h / optimizer.c:
  optimizer interface and simple optimizer implementation

checkpoint.h / checkpoint.c:
  save/load model and optimizer state
```

### `core/runtime/`

Closed-loop policy execution.

```text
policy.h / policy.c:
  loaded inference pipeline

runtime_state.h:
  state for KV cache, recurrent models, Mamba/SSM-style models, action chunks

safety.h:
  safety filter interface

loop.h / loop.c:
  observe, predict, decode, safety-check, act loop
```

### `core/cuda/`

CUDA support and hot kernels.

```text
cuda_common.h:
  CUDA error handling and launch helpers

kernels.h:
  shared kernel declarations

matmul.cu:
  baseline matrix multiplication kernels

softmax.cu:
  softmax and cross-entropy support kernels

attention.cu:
  baseline attention kernels
```

### `projects/line_follower/`

The first concrete project.

```text
line_follower.c:
  line-follower sim/env implementation
  reset, step, raw QTI observation writes, reward, terminals, logs

line_follower.h:
  project state structs and env registration

line_follower.ini:
  first known-good experiment config

README.md:
  project explanation and command examples

docs/:
  project-specific notes if they become large enough to split out
```

### `projects/centerline/`

Tiny deterministic project for tests and learning.

```text
centerline.c:
  integer offset env, oracle, rewards, terminals, logs

centerline.h:
  project state structs and env registration

centerline.ini:
  first known-good full-stack test config

README.md:
  explains the toy task and expected behavior

data/:
  small ignored/generated trajectories or tiny committed fixtures
```

## V0 milestones

1. Write the root architecture and research docs.
2. Define config schema and module spec structs.
3. Implement static module registry and config validation.
4. Implement the `centerline` raw env API.
5. Implement reusable tokenizer/vectorizer modules needed by `centerline`.
6. Implement trajectory log format and project-local data output.
7. Implement a deterministic lookup-table or count-based trainer for `centerline`.
8. Run learned `centerline` policy closed-loop in sim as a test.
9. Add replay tests for tokenizer, decoder, and deterministic env logs.
10. Implement line-follower raw env API.
11. Implement oracle data collector for line follower.
12. Train the simplest line-follower policy that predicts action tokens.
13. Run learned line-follower policy closed-loop in sim.
14. Add profiling counters for env steps/sec and tokens/sec.
15. Add CUDA backends one layer at a time only after CPU reference behavior is deterministic.

## Non-goals for v0

Do not implement every paper method immediately.

Do not build dynamic plugins before static registration works.

Do not deploy to the real robot before sim training and replay are stable.

Do not build a giant generic graph engine.

Do not hide project semantics inside the core.

Do not couple the whole framework to one trainer, one model, or one token format.

## Research references

Humanoid Locomotion as Next Token Prediction:

```text
https://arxiv.org/abs/2402.19469
```

Decision Transformer:

```text
https://arxiv.org/abs/2106.01345
```

Trajectory Transformer:

```text
https://arxiv.org/abs/2106.02039
```

StARformer:

```text
https://arxiv.org/abs/2110.06206
```

Modular paper architecture survey:

```text
docs/research/04-modular-paper-architecture-survey.md
```

PufferLib:

```text
https://github.com/PufferAI/PufferLib
```
