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
- `06-snake-planner-benchmark.md`: Snake lookup-policy failure analysis and the safe planner baseline benchmark.
- `07-loss-layer-options.md`: phase-0 loss layer implementation notes and planned future losses.
- `08-vq-code-tokenizer.md`: fixed-codebook VQ tokenizer notes and planned learned VQ work.
- `09-embedder-layer-options.md`: lookup and linear embedder implementation notes and planned future embedders.
- `10-sequence-layout-layer.md`: typed sequence layout implementation notes and planned future layouts.
- `11-ini-config-layer.md`: minimal INI parser notes and planned layer-selection config work.
- `12-layer-kind-registry.md`: INI layer kind string-to-enum registry and current implementation status.
- `13-head-layer-options.md`: categorical head implementation notes and planned future heads.
- `14-mlp-window-model.md`: one-hidden-layer MLP window model inference notes and future training work.
- `15-ngram-next-token-model.md`: bigram next-token baseline model and its role before transformers.
- `16-continuous-vectorizer.md`: raw and standardized continuous vector path for MLP/VQ/model inputs.
- `17-ngram-benchmarks.md`: ngram next-token benchmark results across centerline, Breakout, and Snake.
- `18-layer-factory.md`: scalar-only INI-driven construction for int bins, raw continuous vectors, and ngram models.
- `19-flat-parameter-store.md`: named flat float arrays for future codebooks, embedding tables, and model weights.
- `20-centerline-mlp-benchmark.md`: first end-to-end factory-built MLP policy benchmark.
- `21-sparse-lookup-policy.md`: exact-token sparse lookup policy and Snake planner-distillation benchmark.
- `22-sparse-lookup-cross-env-benchmarks.md`: sparse lookup benchmark results across Centerline, Breakout, and Snake.
- `23-snake-multistart-distillation.md`: multistart Snake sparse lookup benchmark and limits.
- `24-nearest-policy.md`: continuous nearest-neighbor policy layer and cross-env benchmark results.
- `25-snake-safety-filter.md`: immediate-action safety filter and held-out Snake benchmark.
- `26-linear-policy.md`: CPU-trained multiclass linear policy and first benchmark results.
- `27-mlp-training.md`: first CPU MLP backprop path and Centerline benchmark.
- `28-snake-mlp-policy.md`: Snake continuous-feature MLP policy trained from planner rollouts and evaluated on held-out starts.
- `29-mlp-cross-entropy-training.md`: categorical cross-entropy MLP training path and cross-project benchmark results.
- `30-action-mask-head.md`: reusable masked-argmax action head and Snake no-planner-fallback benchmark.
- `31-masked-cross-entropy-training.md`: masked categorical cross-entropy training and the current negative Snake result.
- `32-dagger-style-training.md`: DAgger-style online planner labeling, improved no-planner Snake benchmark, and 64-hidden capacity negative result.
- `33-snake-action-aware-features.md`: action-aware continuous Snake features and the current best no-planner learned Snake benchmark.
- `34-action-conditioned-scorers.md`: paper-backed next candidate for shared state-action scoring models.
- `35-action-scorer-benchmark.md`: implemented shared action scorer, action-space feature split, and new best Snake benchmark.
- `36-action-scorer-cross-project.md`: action scorer validation across Centerline, Breakout, and Snake.
- `37-action-scorer-config.md`: INI registry/factory support for constructing action scorers from flat params.
- `50-action-feature-config.md`: config-selectable action feature rows and the rejected Snake `food_space` default experiment.
- `51-training-rate-config.md`: config-selectable trainer rates and the neutral Snake learning-rate sweep.
- `52-rollout-score-decoder.md`: short-horizon rollout decoder fallback and the new strongest Snake config benchmark.
- `53-snake-config-runtime-render.md`: project-local runtime helper and Raylib renderer bridge for the configured Snake policy.
- `54-pufferlib-breakout-jsonl-benchmark.md`: PufferLib Breakout JSONL exporter fix, cleaned 100k-row dataset, no-render benchmark, and trained intercept policy result.
- `55-pufferlib-breakout-token-mlp.md`: generic PufferLib Breakout token MLP policy, no-render rollout results, and render default update.
- `56-pufferlib-breakout-layered-token-stack.md`: INI-selected PufferLib Breakout token stack with `mlp_window` as the known-good backend and `linear_policy` as a swappable backend proof.

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
