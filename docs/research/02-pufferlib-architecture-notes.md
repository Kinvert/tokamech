# PufferLib Architecture Notes

Research snapshot: 2026-07-06.

Primary source:

- https://github.com/PufferAI/PufferLib

Paper source:

- https://arxiv.org/abs/2406.12905

## Main takeaway

PufferLib's useful lesson is the flat, fast boundary between core and environments.

Tokamech should adapt:

```text
plain C env ABI
flat contiguous buffers
preallocated memory
explicit ownership
standalone env executables
profiling from day one
```

Tokamech should not blindly copy:

```text
monolithic build scripts
PPO-specific trainer coupling
Python compatibility layers as internal abstractions
algorithm-specific CUDA kernel sequence_layout
```

## Repo shape worth adapting

PufferLib separates:

```text
src/        core runtime and trainer implementation
ocean/      first-party environments
pufferlib/  Python orchestration layer
tests/      API, kernel, benchmark, and profiling tests
```

Tokamech equivalent could be:

```text
core/       C/CUDA runtime, buffers, token sequences, model kernels
projects/   first-party projects such as line_follower
tools/      data conversion, training entrypoints, profiling helpers
docs/       architecture, research, design notes
tests/      unit, replay, kernel-reference, and smoke tests
```

## Core/env boundary

PufferLib's environment contract is intentionally flat:

```text
env owns state
env exposes reset/step/log/render/close
core owns large shared buffers
env reads/writes its assigned slice
```

Tokamech should follow the same principle.

Possible Tokamech env shape:

```c
typedef struct TkmEnv TkmEnv;

typedef struct {
    uint32_t num_envs;
    uint32_t obs_bytes;
    uint32_t action_bytes;
    void* observations;
    void* actions;
    float* rewards;
    uint8_t* dones;
} TkmEnvBuffers;

void tkm_env_reset(TkmEnv* env, uint32_t seed);
void tkm_env_step(TkmEnv* env);
void tkm_env_close(TkmEnv* env);
```

The exact ABI should be designed later, but the principle is clear: avoid object-heavy abstractions in hot loops.

## Vectorization strategy

PufferLib's useful pattern:

```text
many env instances
contiguous agent/env slots
fixed-size rollout buffers
host/GPU transfer boundaries are explicit
training sequence_layout can differ from env-write sequence_layout
```

Tokamech should plan for:

```text
env-major sequence_layout for stepping
sequence-major or batch-major sequence_layout for training
stable pointers for future CUDA graph capture
fixed shapes in hot paths
```

For v0, this can stay simple:

```text
N simulated line-follower envs
T timestep rollout
write token frames into contiguous buffer
train from sampled fixed-length windows
```

## C/CUDA style

Patterns worth adapting:

```text
plain structs
raw pointers
compile-time metadata where useful
small tensor structs instead of heavyweight classes
explicit allocation/free
minimal hidden control flow
```

Tokamech style target:

```text
boring C
boring CUDA
short files with clear jobs
no code golf
no macro maze unless performance forces it
easy to read under picky code review
```

## Build approach

PufferLib uses a central shell build driver with env-specific metadata extraction.

Tokamech should probably prefer a cleaner greenfield build:

```text
CMake or Meson
explicit project metadata files
standalone env binaries
CUDA optional at first if needed
no awk-based macro extraction
```

The core lesson is not the exact build script. The lesson is that each environment should build both:

```text
as a training/runtime plugin
as a standalone debug executable
```

## Testing and profiling lessons

Useful guardrails:

```text
fast import/startup checks
kernel-reference parity tests
microbenchmarks
standalone env smoke tests
replay tests for deterministic logs
SPS and timing metrics in normal output
NVTX ranges for CUDA profiling later
```

For Tokamech v0:

```text
replay a token log through tokenizer and policy
compare C tokenizer against reference data
verify env determinism with fixed seeds
benchmark env steps/sec
benchmark tokens/sec for training
```

## Tokamech implication

Architecture should split into four durable boundaries:

```text
project/env: creates experience
tokenizer: converts project-specific state/actions to typed tokens
dataset: stores sequences and windows
learner/runtime: trains and runs the policy
```

The core should know enough about token types and shapes to batch efficiently. It should not know what a QTI sensor, joystick button, humanoid joint, or Atari frame means.

