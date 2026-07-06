# Tokamech Research Notes

Research snapshot: 2026-07-06.

This folder stores the project research that motivates the architecture. The goal is not to copy any one paper or framework. The goal is to extract the useful ideas for a small, understandable, C/raw-CUDA-first system that can start with a tokenized sim2real line follower and later support more environments.

## Current recommendation

Start with typed discrete tokens:

```text
observation token: compact integer representation of sensor state
action token: compact integer representation of motor/steering command
special tokens: PAD, BOS, EOS, MASK, RESET
```

The first project should prove offline behavior cloning in simulation:

```text
line-follower env -> rollout logs -> token sequences -> train causal model -> run policy in sim
```

Robot deployment should be a follow-up milestone after the token/data/training/runtime loop works in simulation.

## Glossary

`BOS` means beginning of sequence. It is a special marker token placed before the first real item in an episode or training window.

`EOS` means end of sequence. It is a special marker token placed after the final real item in an episode.

`PAD` means padding. It fills unused slots when batching sequences of different lengths.

`MASK` means unknown or intentionally hidden token. It is useful later for missing modalities, corrupted inputs, or auxiliary training tasks.

`RESET` means environment reset boundary. It is useful when a packed stream contains multiple episodes.

## Research files

- `01-tokenized-control-papers.md`: sequence-model and tokenized-control paper notes.
- `02-pufferlib-architecture-notes.md`: PufferLib architecture lessons to adapt.
- `03-line-follower-tokenization.md`: line-follower observation/action/data design.
- `04-modular-paper-architecture-survey.md`: survey of which papers fit the interchangeable-layer architecture and which require broader system changes.
- `05-method-glossary-and-learning-map.md`: beginner-facing explanation of the paper/method names and how to study each layer.

## Core architectural opinion

The repo should treat control as sequences, but it should not force every project into one global vocabulary.

Good v0 shape:

```text
project tokenizer owns obs/action vocabularies
core owns sequence buffers, batching, model execution, losses, and runtime loops
env owns reset/step/log/render and raw simulation state
```

For line following, the first token stream can be:

```text
BOS
OBS(sensor_pattern_t0)
PREV_ACTION(straight)
ACTION(slight_left)
OBS(sensor_pattern_t1)
PREV_ACTION(slight_left)
ACTION(left)
EOS
```

The first model should predict the next action token from recent observation/action history. Auxiliary next-observation prediction can be added later.
