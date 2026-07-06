# Modular Paper Architecture Survey

Survey date: July 2026.

Tokamech's proposed stack is a good fit for many sequence-model control papers, but not for all of robot learning. The most compatible papers can be studied by swapping one layer at a time: action tokenization, layout, sequence backbone, head, loss, trainer, or decoder. The least compatible papers change the whole system shape: they introduce online data collection, latent world models, planning loops, foundation VLM pretraining, or runtime constraints that are not just another module behind the same interface.

Tokamech layers used below:

- `sequence_layout`: ordering of observations, actions, rewards, returns, goals, language, video frames, modalities, chunks, and horizons.
- `tokenizer/vectorizer`: conversion of raw state, image, action, reward, proprioception, command, or text into discrete tokens or continuous vectors.
- `embedder`: projection of those vectors/tokens into model width, plus modality/type/position embeddings.
- `sequence_model`: Transformer, Mamba/RWKV/state-space model, temporal convolution, recurrent model, or other sequence backbone.
- `prediction_head`: logits, continuous regression head, mixture head, value head, dynamics head, diffusion noise head, etc.
- `loss`: cross-entropy, MSE, action reconstruction, offset loss, diffusion denoising, TD loss, world-model losses.
- `trainer`: offline BC, return-conditioned imitation, RL fine-tuning, world-model training, multi-task pretraining, online collection.
- `decoder`: mapping model outputs back to executable actions, action chunks, beam-search plans, receding-horizon controls, or denoised action trajectories.
- `data_collector/runtime`: teleoperation, robot rollouts, simulator interaction, online replay, planning-time search, latency and control-rate machinery.

## Summary table

| Fit | Paper / family | Main idea | Primary Tokamech layers touched | Why it fits or does not |
|---|---|---|---|---|
| Strong | [Decision Transformer](https://arxiv.org/abs/2106.01345) | Treat offline RL as return-conditioned sequence modeling. | `sequence_layout`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer` | Mostly a clean layout and conditioning change: `(return, state, action)` tokens with causal action prediction. |
| Strong | [Trajectory Transformer](https://arxiv.org/abs/2106.02039) | Model whole trajectories and use beam search for planning. | `sequence_layout`, `tokenizer/vectorizer`, `sequence_model`, `prediction_head`, `decoder` | Natural sequence-model core, but planning decoder is important and should be explicit. |
| Strong | [Behavior Transformer / BeT](https://arxiv.org/abs/2206.11251) | Discretize multimodal continuous actions, then predict action clusters plus residual offsets. | `tokenizer/vectorizer`, `prediction_head`, `loss`, `decoder` | Excellent layer-isolation example: most novelty lives in action tokenization/head/loss/decoder. |
| Strong | [VQ-BeT](https://arxiv.org/abs/2403.03181) | Replace k-means-style action modes with hierarchical vector quantization for latent action generation. | `tokenizer/vectorizer`, `prediction_head`, `loss`, `decoder` | Even cleaner for Tokamech than BeT: a learnable action tokenizer can be swapped independently. |
| Strong | [ACT / ALOHA](https://arxiv.org/abs/2304.13705) | Predict short future action chunks for imitation learning and smooth them at runtime. | `sequence_layout`, `prediction_head`, `loss`, `decoder`, `data_collector/runtime` | Strong if Tokamech treats chunking and temporal ensembling as decoder/runtime modules, not hidden policy code. |
| Strong | [FAST](https://arxiv.org/abs/2501.09747) | Frequency-space action sequence tokenization for autoregressive VLA policies. | `tokenizer/vectorizer`, `decoder`, `loss` | Almost exactly a tokenizer/decoder study case: swap per-dimension bins for DCT/compression action tokens. |
| Strong | Mamba/RWKV-style sequence backbones, e.g. [Decision Mamba](https://arxiv.org/abs/2403.19925) and [Mamba](https://arxiv.org/abs/2312.00752) | Replace attention with selective state-space or recurrent-style sequence modeling. | `sequence_model`, sometimes `trainer` and `runtime` | Good one-layer swap if the input/output token contract stays fixed; runtime state/cache interface needs care. |
| Partial | [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469) | Autoregressively predict multimodal sensorimotor tokens for humanoid locomotion, using missing-modality-friendly training. | `sequence_layout`, `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `decoder`, `data_collector/runtime` | Philosophically aligned, but it coordinates multimodal layout, modality-specific prediction heads, data mixture, and robot runtime. |
| Partial | [StARformer](https://arxiv.org/abs/2110.06206) | Add local state-action-reward representations before long-horizon attention. | `sequence_layout`, `embedder`, `sequence_model` | Decomposable, but the layout and model are intertwined through short-window StAR representation blocks. |
| Partial | [RT-1](https://arxiv.org/abs/2212.06817) | Large-scale real-robot Transformer policy trained on diverse tasks. | `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `data_collector/runtime` | Modular inside the model, but the contribution depends heavily on dataset scale, embodiment/action spec, and robot deployment. |
| Partial | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Generate action trajectories through conditional denoising diffusion with receding-horizon control. | `prediction_head`, `loss`, `trainer`, `decoder`, `sequence_layout` | Fits if `decoder` and `loss` are flexible enough for iterative denoising; not a simple next-token head. |
| Poor / paradigm shift | [RT-2](https://arxiv.org/abs/2307.15818) | Co-fine-tune pretrained VLMs so robot actions are emitted as text-like tokens. | `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime` | It is a VLM adaptation recipe more than a single control-module change; web-scale pretraining dominates the system shape. |
| Poor / paradigm shift | [TD-MPC](https://arxiv.org/abs/2203.04955) | Learn latent dynamics and terminal values for short-horizon model predictive control. | `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime` | Requires a model-based RL loop: latent dynamics, value learning, planner, replay, and online data collection are coupled. |
| Poor / paradigm shift | [DreamerV3](https://arxiv.org/abs/2301.04104) | Learn a world model and train actor/critic behavior by imagination. | `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime` | Valuable to study, but the agent is a world-model system, not just sequence-in/action-out supervised control. |

## Strong fit: mostly one layer or adjacent layers

### Decision Transformer

[Decision Transformer](https://arxiv.org/abs/2106.01345) is the canonical strong fit. It changes the framing of offline RL into causal sequence modeling by arranging return-to-go, state, and action into a token stream and training the model to predict actions.

Primary layers: `sequence_layout`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`.

Tokamech lesson: start with a fixed continuous-state vectorizer and fixed Transformer, then swap only `sequence_layout`: behavior cloning layout versus `(return, state, action)` layout. This is a clean educational path because the loss can remain supervised action prediction.

Limitation: the method depends on good return conditioning and offline datasets. It does not solve exploration or online data collection.

### Trajectory Transformer

[Trajectory Transformer](https://arxiv.org/abs/2106.02039) models trajectories as sequences and uses beam search as a planning decoder. It is a strong fit if Tokamech treats planning as a first-class `decoder`, not as part of the model internals.

Primary layers: `sequence_layout`, `tokenizer/vectorizer`, `sequence_model`, `prediction_head`, `decoder`.

Tokamech lesson: keep the same autoregressive model but compare greedy action decoding, beam search over future states/actions/rewards, and constrained planning. This teaches why a decoder is not just argmax.

Limitation: discretizing continuous states/actions and planning over generated trajectories can be brittle. The paper is not merely a policy-head swap.

### Behavior Transformer and VQ-BeT

[Behavior Transformer](https://arxiv.org/abs/2206.11251) and [VQ-BeT](https://arxiv.org/abs/2403.03181) are excellent examples of action-interface research. BeT uses action discretization plus residual correction for multimodal continuous actions. VQ-BeT replaces the action-mode representation with hierarchical vector quantization.

Primary layers: `tokenizer/vectorizer`, `prediction_head`, `loss`, `decoder`.

Tokamech lesson: use the same observations, same sequence backbone, and same trainer; swap the action representation. Compare direct MSE regression, k-means action bins plus offsets, and VQ action latents.

Limitation: action tokenizers are not universal. A tokenizer that works for low-dimensional end-effector deltas may fail for high-frequency dexterous hands unless it represents temporal structure.

### ACT / ALOHA

[Action Chunking with Transformers](https://arxiv.org/abs/2304.13705), introduced with ALOHA, predicts chunks of future actions and combines overlapping predictions during execution. This is a strong fit if Tokamech separates chunk prediction from chunk execution.

Primary layers: `sequence_layout`, `prediction_head`, `loss`, `decoder`, `data_collector/runtime`.

Tokamech lesson: hold the visual/proprioceptive encoder fixed, then compare single-step BC against chunked prediction. Learners can see compounding error, latency, and smoothing as concrete runtime phenomena.

Limitation: ACT is partly an algorithm and partly a data-collection/control-rate recipe. The teleoperation setup and temporal ensembling matter.

### FAST and action-tokenizer work

[FAST](https://arxiv.org/abs/2501.09747) is one of the cleanest fits for Tokamech's layer philosophy. It argues that action tokenization is a bottleneck for autoregressive VLA policies and replaces simple per-dimension binning with frequency-space action sequence tokenization. Newer action-tokenizer work such as [X-Tokenizer](https://arxiv.org/abs/2606.14752) pushes the same boundary toward semantic action interfaces.

Primary layers: `tokenizer/vectorizer`, `decoder`, `loss`.

Tokamech lesson: teach action tokenization like MNIST teaches pixels. Start with scalar bins, then chunk bins, then VQ, then DCT/frequency tokens. Keep the model and trainer fixed so reconstruction error, action smoothness, and policy success can be attributed to the tokenizer.

Limitation: a learned or compressed tokenizer can hide control assumptions. Tokamech should expose reconstruction metrics and action playback before policy training.

### Mamba, RWKV, and state-space backbones

[Mamba](https://arxiv.org/abs/2312.00752)-style models and recurrent alternatives such as RWKV are strong fits when the token contract stays unchanged. [Decision Mamba](https://arxiv.org/abs/2403.19925) is a direct example: keep the Decision Transformer framing and swap the sequence backbone.

Primary layers: `sequence_model`, sometimes `trainer` and `data_collector/runtime`.

Tokamech lesson: this is the cleanest way to study sequence-model kernels in C/CUDA. Run the same dataset and layout through causal attention, recurrent state, SSM scan, and KV-cache-style inference.

Limitation: the interface must include runtime state. A stateless Transformer forward and a recurrent/SSM streaming forward should share a high-level contract but not pretend to have identical memory behavior.

## Partial fit: decomposable, but coordinated modules matter

### Humanoid Locomotion as Next Token Prediction

[Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469) is philosophically close to Tokamech. It casts humanoid control as autoregressive prediction over multimodal sensorimotor trajectories and handles missing modalities by predicting the next token from the same modality.

Primary layers: `sequence_layout`, `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `decoder`, `data_collector/runtime`.

Tokamech lesson: this is a capstone integration target after individual layers are understood. It combines modality layout, missing-modality masks, sensor/action tokenization, autoregressive loss, and real-time control.

Limitation: it should not be presented as just "Decision Transformer for humanoids." The data mixture, modality alignment, missing modalities, and deployment loop are central.

### StARformer

[StARformer](https://arxiv.org/abs/2110.06206) builds local state-action-reward representations before longer-horizon attention, adding a Markov-like short-window inductive bias.

Primary layers: `sequence_layout`, `embedder`, `sequence_model`.

Tokamech lesson: show that `embedder` can be more than a linear projection. It can include local cross-token interaction before global sequence modeling.

Limitation: the boundary between embedder and model becomes blurry. If Tokamech wants clean pedagogical modules, StARformer should be implemented as a composite `embedder + sequence_model` example, not forced into one box.

### RT-1

[RT-1](https://arxiv.org/abs/2212.06817) is a partial fit. The model itself can be decomposed into visual tokenization, language/task conditioning, Transformer sequence processing, and action prediction. But much of the result comes from large-scale robot data collection and task diversity.

Primary layers: `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `data_collector/runtime`.

Tokamech lesson: useful for studying generalist policy interfaces: image tokens, language commands, action discretization, and multi-task batches.

Limitation: a small educational implementation will teach the architecture but not reproduce the scaling claim. Tokamech should label this clearly.

### Diffusion Policy

[Diffusion Policy](https://arxiv.org/abs/2303.04137) is modular but not next-token modular. The policy predicts denoising steps over action trajectories and uses receding-horizon execution.

Primary layers: `prediction_head`, `loss`, `trainer`, `decoder`, `sequence_layout`.

Tokamech lesson: compare explicit action-token decoding with iterative continuous-action decoding. This is useful for teaching why multimodal actions are hard for MSE regression.

Limitation: diffusion changes the training target, inference loop, and runtime budget together. It should be a supported alternative decoder/loss family, not squeezed into a cross-entropy token head.

## Poor fit / paradigm shift: useful, but not one-layer-swappable

### RT-2 and VLA foundation-model adaptation

[RT-2](https://arxiv.org/abs/2307.15818) adapts pretrained vision-language models to output robot actions as tokens. It is important, but it is a poor fit for a small, raw-CUDA-first modular control stack if treated as a normal method swap.

Primary layers: `tokenizer/vectorizer`, `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime`.

Tokamech lesson: RT-2 is best studied as a boundary case. It shows what happens when the `sequence_model` is no longer a local research component but a pretrained VLM with its own tokenizer, optimizer assumptions, data scale, and serving constraints.

Limitation: action-as-language-token is modular on paper, but the main contribution depends on web-scale pretraining and co-fine-tuning. Tokamech can emulate the interface, not the full paradigm.

### TD-MPC

[TD-MPC](https://arxiv.org/abs/2203.04955) learns task-oriented latent dynamics and a terminal value function for model predictive control. This is not a simple sequence policy. It couples representation learning, dynamics prediction, value learning, planning, and online replay.

Primary layers: `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime`.

Tokamech lesson: useful for showing where sequence-model control ends and model-based control begins. It motivates a separate `planner/world_model` extension rather than overloading `decoder`.

Limitation: one-layer swapping is misleading here. Change the latent model and the planner/loss/trainer all move with it.

### Dreamer / world-model agents

[DreamerV3](https://arxiv.org/abs/2301.04104) learns a world model and trains actor/critic behavior through imagined rollouts. Like TD-MPC, it is valuable but cuts across the whole stack.

Primary layers: `embedder`, `sequence_model`, `prediction_head`, `loss`, `trainer`, `decoder`, `data_collector/runtime`.

Tokamech lesson: Dreamer is a good future curriculum chapter after supervised sequence control. It teaches latent dynamics, reconstruction or representation prediction, actor-critic losses, replay, and online interaction.

Limitation: it requires more than the proposed layer list. Tokamech would need explicit modules for `world_model`, `actor`, `critic`, `replay_buffer`, and `imagined_rollout` to represent it honestly.

## Educational path: studying one layer at a time

Tokamech can be an educational framework if each experiment changes one thing and logs the right failure modes.

1. `sequence_layout`: Begin with behavior cloning on `(state -> action)`. Then add history windows, then Decision Transformer `(return, state, action)`, then Trajectory Transformer full-trajectory layout. Keep vectorizer/model/head fixed.
2. `tokenizer/vectorizer`: Compare continuous state normalization, scalar action bins, k-means action bins, VQ action tokens, and FAST-style chunk/frequency tokens. Measure reconstruction before policy success.
3. `embedder`: Start with linear projections and learned modality embeddings. Add image patch embeddings, proprioception embeddings, action/reward type embeddings, and StARformer-style local fusion.
4. `sequence_model`: Swap causal Transformer, tiny recurrent model, Mamba/SSM, and RWKV-like recurrence under the same token contract. Study memory, latency, and long-context behavior.
5. `prediction_head`: Compare MSE regression, categorical logits, mixture/residual heads from BeT, chunk heads from ACT, and diffusion noise heads.
6. `loss`: Teach cross-entropy, MSE, negative log-likelihood, offset losses, VQ commitment/reconstruction losses, diffusion denoising losses, and TD/world-model losses separately.
7. `trainer`: Keep architecture fixed while changing offline BC, return-conditioned BC, multi-task training, data balancing, online fine-tuning, and replay.
8. `decoder`: Compare greedy next action, action detokenization, residual correction, temporal ensembling, beam search, receding-horizon action chunks, and diffusion denoising.
9. `data_collector/runtime`: Treat this as a real module, not an afterthought. Control frequency, latency, action smoothing, observation delay, safety clamps, and teleoperation quality can dominate robotics results.

A good curriculum sequence is:

1. Direct BC with continuous MSE actions.
2. Decision Transformer layout with the same model.
3. BeT/VQ-BeT action tokenizer with the same model.
4. ACT chunk decoder with the same observations.
5. Trajectory Transformer planning decoder.
6. Mamba/SSM backbone swap.
7. Diffusion Policy as an alternate loss/head/decoder family.
8. RT-1-style multi-modal conditioning.
9. Humanoid next-token prediction as a capstone.
10. TD-MPC/Dreamer as a separate model-based track.

## Recommendations for Tokamech boundaries

Keep these boundaries stable:

- Stable sample schema: observations, actions, rewards, returns, terminals, timesteps, task ids, language, modality masks, and robot/action metadata should have a stable in-memory representation.
- Stable token contract: every `tokenizer/vectorizer` should declare token type, shape, rate, vocabulary or continuous dimension, valid mask, and inverse decode capability when applicable.
- Stable sequence layout API: layouts should transform episodes into training sequences without owning model code.
- Stable model I/O: `sequence_model` should consume embedded sequences plus masks and return hidden states. It should not know whether actions came from bins, VQ codes, or FAST tokens.
- Stable head/decoder split: `prediction_head` produces model-space outputs; `decoder` turns those outputs into executable controls or plans.
- Stable runtime state interface: streaming models, KV-cache Transformers, Mamba/SSM state, and action-chunk decoders need explicit state objects.
- Stable metrics hooks: reconstruction error, token entropy, action smoothness, rollout success, latency, and memory should be comparable across layer swaps.

Keep these flexible:

- Action representation: direct continuous, scalar bins, residual bins, VQ latents, DCT/frequency tokens, diffusion trajectories, and flow outputs should all be possible.
- Modality layout: robotics papers disagree on whether images, proprioception, language, rewards, returns, and actions are separate streams, interleaved tokens, or fused embeddings.
- Decoder semantics: Tokamech should support argmax, sampling, beam search, temporal ensembling, receding-horizon chunks, and iterative denoising.
- Loss composition: many papers need multiple losses. The framework should allow supervised, reconstruction, commitment, denoising, TD, and auxiliary prediction losses without rewriting the trainer.
- Trainer/data loop: offline imitation, offline RL, online RL, world-model training, and VLA fine-tuning should not be forced into one trainer abstraction.
- Runtime control assumptions: control frequency, latency, safety clipping, action hold, smoothing, and observation synchronization must remain configurable.

The practical recommendation is to make Tokamech's core pedagogical path sequence-first and supervised-first, because Decision Transformer, BeT, VQ-BeT, ACT, FAST, Trajectory Transformer, and Mamba-style swaps all fit that path. Do not overgeneralize the same abstraction to RT-2, TD-MPC, or Dreamer. Instead, mark those as extension tracks that deliberately add foundation-model or world-model system components.

## Source links

- [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/abs/2402.19469)
- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- [Offline Reinforcement Learning as One Big Sequence Modeling Problem / Trajectory Transformer](https://arxiv.org/abs/2106.02039)
- [Behavior Transformers: Cloning k modes with one stone](https://arxiv.org/abs/2206.11251)
- [Behavior Generation with Latent Actions / VQ-BeT](https://arxiv.org/abs/2403.03181)
- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [StARformer: Transformer with State-Action-Reward Representations for Visual Reinforcement Learning](https://arxiv.org/abs/2110.06206)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT / ALOHA](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Temporal Difference Learning for Model Predictive Control / TD-MPC](https://arxiv.org/abs/2203.04955)
- [Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2403.19925)
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
- [X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining](https://arxiv.org/abs/2606.14752)
