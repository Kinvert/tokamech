# Benchmarks

Date: 2026-07-06

Run benchmarks headlessly:

```sh
make benchmark
```

Do not use `DISPLAY=:0` for benchmark runs.

## Current benchmark output

```text
centerline steps=8 score=1 metric=1
centerline_ngram steps=8 score=1 metric=1
centerline_mlp steps=8 score=1 metric=1
centerline_mlp_trained steps=8 score=1 metric=1
centerline_mlp_ce_trained steps=8 score=1 metric=1
centerline_mlp_masked_ce_trained steps=8 score=1 metric=1
centerline_sparse_lookup steps=8 score=1 metric=1
centerline_nearest steps=8 score=1 metric=1
centerline_linear_policy steps=8 score=1 metric=1
centerline_action_scorer steps=8 score=1 metric=1
centerline_action_scorer_config_file steps=8 score=1 metric=1
breakout steps=220 score=2 metric=3
breakout_ngram steps=220 score=2 metric=3
breakout_sparse_lookup steps=220 score=2 metric=3
breakout_nearest steps=220 score=2 metric=3
breakout_linear_policy steps=220 score=2 metric=3
breakout_action_scorer steps=220 score=2 metric=3
breakout_action_scorer_config_file steps=220 score=2 metric=3
breakout_mlp_ce_trained steps=220 score=2 metric=3
breakout_mlp_masked_ce_trained steps=220 score=2 metric=3
snake_lookup steps=3 score=1 metric=1
snake_ngram steps=3 score=1 metric=1
snake_sparse_lookup steps=96 score=11 metric=11
snake_sparse_lookup_multistart steps=288 score=34 metric=34
snake_nearest_multistart steps=288 score=34 metric=34
snake_nearest_safe_holdout steps=480 score=47 metric=47
snake_mlp_safe_holdout steps=480 score=46 metric=46
snake_mlp_ce_safe_holdout steps=480 score=48 metric=48
snake_mlp_ce_masked_holdout steps=480 score=30 metric=30
snake_mlp_masked_ce_holdout steps=439 score=17 metric=17
snake_mlp_dagger_masked_holdout steps=480 score=40 metric=40
snake_mlp_dagger64_masked_holdout steps=480 score=28 metric=28
snake_mlp_action_features_holdout steps=480 score=51 metric=51
snake_action_scorer_holdout steps=480 score=58 metric=58
snake_action_scorer_ce_holdout steps=480 score=58 metric=58
snake_action_scorer_config_margin_holdout steps=480 score=58 metric=58
snake_action_scorer_config_ce_holdout steps=480 score=58 metric=58
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
snake_planner steps=96 score=11 metric=11
```

## Interpretation

`centerline`

```text
metric = 1 means the policy reaches zero offset.
```

`breakout`

```text
metric = paddle hits during the rollout.
```

`centerline_sparse_lookup`

```text
metric = 1 means sparse exact-token imitation reaches zero offset.
```

`centerline_nearest`

```text
metric = 1 means continuous nearest-neighbor imitation reaches zero offset.
```

`centerline_linear_policy`

```text
metric = 1 means a CPU-trained linear classifier reaches zero offset.
```

`centerline_action_scorer`

```text
metric = 1 means the shared action scorer reaches zero offset.
```

This benchmark proves the action scorer can work outside Snake on a minimal control problem.

`centerline_action_scorer_config_file`

```text
metric = 1 means the project-local Centerline action-scorer INI selects the shared scorer and reaches zero offset.
```

`centerline_mlp`

```text
metric = 1 means the fixed-weight factory-built MLP policy reaches zero offset.
```

`centerline_mlp_trained`

```text
metric = 1 means a CPU-trained MLP reaches zero offset.
```

`centerline_mlp_ce_trained`

```text
metric = 1 means a CPU-trained MLP using categorical cross entropy reaches zero offset.
```

`centerline_mlp_masked_ce_trained`

```text
metric = 1 means a CPU-trained MLP using masked categorical cross entropy reaches zero offset.
```

`centerline_ngram`

```text
metric = 1 means the ngram next-token policy reaches zero offset.
```

`breakout_ngram`

```text
metric = paddle hits during the ngram next-token rollout.
```

`breakout_sparse_lookup`

```text
metric = paddle hits during sparse exact-token imitation rollout.
```

`breakout_nearest`

```text
metric = paddle hits during continuous nearest-neighbor imitation rollout.
```

`breakout_linear_policy`

```text
metric = paddle hits during CPU-trained linear classifier rollout.
```

`breakout_action_scorer`

```text
metric = paddle hits during shared action-scorer rollout.
```

The action scorer evaluates each candidate paddle move by how close the next paddle center is to the ball center.

`breakout_action_scorer_config_file`

```text
metric = paddle hits when the project-local Breakout action-scorer INI selects the shared scorer.
```

`breakout_mlp_ce_trained`

```text
metric = paddle hits during CPU-trained MLP categorical cross-entropy rollout.
```

`breakout_mlp_masked_ce_trained`

```text
metric = paddle hits during CPU-trained MLP masked categorical cross-entropy rollout.
```

`snake_lookup`

```text
metric = food eaten by the compact-token lookup policy.
```

This is intentionally kept as a weak baseline.

`snake_sparse_lookup`

```text
metric = food eaten by a sparse full-state lookup policy distilled from planner rollouts.
```

This is the first learned Snake policy that plays the full benchmark well. It is still imitation over a deterministic planner rollout, not a general neural policy.

`snake_sparse_lookup_multistart`

```text
metric = total food eaten across three 96-step deterministic first-food starts.
```

This is a stronger Snake benchmark than the single fixed start. It still trains on planner rollouts for those starts, so it measures multi-scenario distillation, not unseen-state generalization.

`snake_nearest_multistart`

```text
metric = total food eaten by continuous nearest-neighbor imitation across three deterministic starts.
```

This uses continuous Snake features instead of exact full-state token equality. The current benchmark still evaluates trained starts, not held-out random starts.

`snake_nearest_safe_holdout`

```text
metric = total food eaten by nearest-neighbor actions plus immediate safety filtering across five held-out first-food starts.
```

The learned nearest policy proposes actions. The Snake safety filter only replaces actions that would terminal on the next step. This is a held-out benchmark, but it is not pure learned-policy generalization.

`snake_mlp_safe_holdout`

```text
metric = total food eaten by a CPU-trained MLP policy plus immediate safety filtering across five held-out first-food starts.
```

The MLP is trained from planner rollouts using continuous Snake features. The five evaluated first-food starts are excluded from training.

`snake_mlp_ce_safe_holdout`

```text
metric = total food eaten by a CPU-trained MLP using categorical cross entropy plus immediate safety filtering across five held-out first-food starts.
```

This currently outperforms the MSE-trained MLP on the same held-out starts.

`snake_mlp_ce_masked_holdout`

```text
metric = total food eaten by a CPU-trained MLP using categorical cross entropy and masked argmax across five held-out first-food starts.
```

The action mask blocks immediately terminal actions before argmax. Unlike `snake_mlp_ce_safe_holdout`, this benchmark does not use the planner as a fallback action provider during evaluation.

`snake_mlp_masked_ce_holdout`

```text
metric = total food eaten by a CPU-trained MLP using masked categorical cross entropy during training and masked argmax during evaluation.
```

This currently underperforms unmasked CE training plus masked argmax. That negative result is useful: the current planner labels already avoid most invalid actions, and strict masked training produces weaker Snake imitation with this tiny MLP.

`snake_mlp_dagger_masked_holdout`

```text
metric = total food eaten by a CPU-trained MLP using DAgger-style online planner labeling and masked argmax during evaluation.
```

This improves the no-planner masked policy from `30` to `40` food on the same five held-out first-food starts. It still trails planner-fallback safety, but it is the best current no-planner learned Snake policy.

`snake_mlp_dagger64_masked_holdout`

```text
metric = total food eaten by a 64-hidden-unit CPU-trained MLP using DAgger-style online planner labeling and masked argmax during evaluation.
```

This currently underperforms the 32-hidden-unit DAgger benchmark. More capacity alone is not the next obvious Snake improvement.

`snake_mlp_action_features_holdout`

```text
metric = total food eaten by a CPU-trained MLP using action-aware continuous features, DAgger-style online planner labeling, and masked argmax during evaluation.
```

The feature vector keeps the normal Snake state features and appends four explicit values for each candidate action: immediate validity, food-distance delta, next-row food delta, and next-column food delta. This is the best current no-planner-fallback learned Snake policy.

`snake_action_scorer_holdout`

```text
metric = total food eaten by a shared action-conditioned scorer using state features plus per-candidate action-space features.
```

This scorer evaluates one candidate action at a time with shared weights. Its action-space representation adds reachable-space fraction after the candidate action, which helps avoid short-term valid moves that trap the snake. This is the best current no-planner-fallback learned Snake policy.

`snake_action_scorer_ce_holdout`

```text
metric = total food eaten by the same shared action-conditioned scorer trained with masked softmax cross entropy.
```

This swaps the trainer from pairwise margin updates to a valid-action softmax loss. It ties the margin scorer at `58`, which is a useful positive result: the model, representation, masking, and benchmark path stay interchangeable when the training loss changes.

`snake_action_scorer_config_margin_holdout`

```text
metric = total food eaten by the shared action scorer when `[train] loss = margin` is parsed from INI text.
```

This benchmark proves the margin trainer can be selected by config text instead of only by a hard-coded benchmark function.

`snake_action_scorer_config_ce_holdout`

```text
metric = total food eaten by the shared action scorer when `[train] loss = masked_cross_entropy` is parsed from INI text.
```

This benchmark proves the masked cross-entropy trainer can be selected by the same config seam.

`snake_action_scorer_config_file_holdout`

```text
metric = total food eaten by the shared action scorer when the INI is loaded from the Snake project folder.
```

This is the first benchmark row driven by a real project-local INI file.

`snake_planner`

```text
metric = food eaten by the safe planner policy.
```

This is the current playable Snake baseline.

## Regression rule

Before changing a core layer or project policy, run:

```sh
make test
make benchmark
make build/snake_render
```

Expected current behavior:

```text
all tests pass
centerline metric remains 1
centerline_ngram metric remains 1
centerline_mlp metric remains 1
centerline_mlp_trained metric remains 1
centerline_mlp_ce_trained metric remains 1
centerline_mlp_masked_ce_trained metric remains 1
centerline_sparse_lookup metric remains 1
centerline_nearest metric remains 1
centerline_linear_policy metric remains 1
centerline_action_scorer metric remains 1
centerline_action_scorer_config_file metric remains 1
breakout metric remains at least 3
breakout_ngram metric remains at least 3
breakout_sparse_lookup metric remains at least 3
breakout_nearest metric remains at least 3
breakout_linear_policy metric remains at least 3
breakout_action_scorer metric remains at least 3
breakout_action_scorer_config_file metric remains at least 3
breakout_mlp_ce_trained metric remains at least 3
breakout_mlp_masked_ce_trained metric remains at least 3
snake_lookup remains at least 1
snake_ngram remains at least 1
snake_sparse_lookup remains at least 11 over 96 steps
snake_sparse_lookup_multistart remains at least 34 over 288 aggregate steps
snake_nearest_multistart remains at least 34 over 288 aggregate steps
snake_nearest_safe_holdout remains at least 47 over 480 aggregate steps
snake_mlp_safe_holdout remains at least 20 over 480 aggregate steps
snake_mlp_ce_safe_holdout remains at least 20 over 480 aggregate steps
snake_mlp_ce_masked_holdout remains at least 20 over 480 aggregate steps
snake_mlp_masked_ce_holdout remains at least 17 over 439 aggregate steps
snake_mlp_dagger_masked_holdout remains at least 40 over 480 aggregate steps
snake_mlp_dagger64_masked_holdout remains at least 28 over 480 aggregate steps
snake_mlp_action_features_holdout remains at least 51 over 480 aggregate steps
snake_action_scorer_holdout remains at least 58 over 480 aggregate steps
snake_action_scorer_ce_holdout remains at least 58 over 480 aggregate steps
snake_action_scorer_config_margin_holdout remains at least 58 over 480 aggregate steps
snake_action_scorer_config_ce_holdout remains at least 58 over 480 aggregate steps
snake_action_scorer_config_file_holdout remains at least 58 over 480 aggregate steps
snake_planner remains at least 11 over 96 steps
snake renderer builds headlessly
```

Renderer build is allowed headlessly. Visual runs with `DISPLAY=:0` require explicit user approval.
