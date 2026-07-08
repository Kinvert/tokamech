# Shortcut Audit

## Removed in this pass

- `projects/snake/snake.c`: removed decoder/runtime action replacement helpers that selected planner, best-score, or rollout-score actions instead of executing the model-selected action.
- `core/config/decoder_config.c`: decoder fallback config now accepts only `none`; `planner_score`, `best_score`, and `rollout_score` are rejected.
- `apps/benchmark.c`: removed Snake benchmark entry points whose evaluation depended on planner/safety fallback helpers: nearest safe holdout, MLP safe holdout, and MLP CE safe holdout.
- `projects/snake/action_scorer.ini`: changed decoder fallback from `rollout_score` to `none`.
- `projects/snake/snake.c`: Snake action masks no longer encode terminal, collision, or survival outcomes. All actions are exposed to the policy; bad actions now terminate through normal environment dynamics.
- `projects/snake/snake.c`: Snake action-feature rows no longer encode terminal status, food-distance delta, reachable-space fraction, or immediate food capture. They now encode action identity only.
- `apps/benchmark.c`: default benchmark output now reports unmasked Snake MLP holdouts instead of masked/safety-gated Snake MLP holdouts.
- `apps/benchmark.c`: removed legacy Snake action-feature and action-scorer benchmark functions from active benchmark source.
- `tests/test_shortcut_inventory.c`: added a source-inventory regression so removed Snake shortcut benchmark symbols cannot quietly return.

## Questionable remaining shortcuts

- `core/model/mlp_window.*` remains available as a generic categorical MLP backend because PufferLib `token_mlp_window` uses it for full-vocabulary next-token prediction. It should not be reintroduced as a direct observation-to-action policy surface.
- `apps/benchmark.c`: Centerline, Breakout, and Snake benchmark data now comes from deterministic exploration curricula plus sampled-return weighting. These scripts do not inspect action scores, death/safety outcomes, path plans, or oracle labels, but they are still hand-authored curricula over small deterministic starts rather than broad stochastic self-play.
- `tests/test_benchmark.c`: benchmark tests now require positive metrics for active tokenized lookup, n-gram, and sparse-lookup default entries. These prove the tiny deterministic benchmark contracts, not broad game competence.

## 2026-07-08 additional removals

- Removed the default benchmark app surfaces for `centerline_action_scorer` and `breakout_action_scorer`; they exposed hand-coded action selection paths rather than end-to-end tokenized policy behavior.
- Restricted `apps/pufferlib_breakout_benchmark.c` to token-policy evaluation methods. The former app-level non-token MLP, nearest-window, sequence-cursor, and intercept benchmark methods are no longer exposed from that benchmark binary.
- Added shortcut inventory coverage for these benchmark-surface regressions so the removed non-token paths cannot silently return to the active benchmark apps.

- Token-policy labels such as `token_mlp_all` remain acceptable because they evaluate tokenized policy behavior; the removed shortcut was the standalone non-token `mlp_all` benchmark method.

## 2026-07-08 PufferLib app cleanup

- Restricted `apps/pufferlib_breakout_policy_render.c` to token-policy inference only. The render app no longer exposes direct MLP, nearest-window, sequence-cursor, intercept-rule, or intercept-trained runtime modes.
- Changed `apps/pufferlib_breakout_smoke.c` so its headless evaluation trains and runs `BreakoutPufferlibTokenPolicy` instead of saving/loading/evaluating the direct observation-to-action policy.
- Extended `tests/test_shortcut_inventory.c` to reject reintroducing those direct PufferLib action-policy calls in the active benchmark, render, and smoke apps.


## 2026-07-08 PufferLib library cleanup

- Removed the direct observation-to-action `BreakoutPufferlibPolicy` API and its BC train/save/load/predict tests from `projects/breakout/pufferlib_policy.{c,h}`.
- Removed nearest-neighbor, sequence-cursor replay, and intercept-policy utilities from `projects/breakout/pufferlib_policy.{c,h}` and their tests. These were non-token baselines and should not be available as supported policy paths.
- `tests/test_shortcut_inventory.c` now rejects reintroducing those non-token PufferLib policy APIs in the library, active smoke app, render app, or benchmark app.

## 2026-07-08 PufferNet data-source cleanup

- Removed PufferNet-backed demo data export from `apps/pufferlib_breakout_smoke.c`; the smoke dataset now comes from PufferLib environment transitions driven by a deterministic exploration action stream rather than model weights.
- Removed the `pufferlib-breakout-checkpoint-export` Makefile surface and deleted `apps/pufferlib_breakout_checkpoint_export.c`, which exported actions from PufferNet checkpoint weights.
- `tests/test_shortcut_inventory.c` now rejects reintroducing PufferNet headers, checkpoint export targets, or `breakout_weights.bin` dependencies into active PufferLib data paths.

## 2026-07-08 Snake oracle/planner cleanup

- Removed `snake_oracle_action`, `snake_planner_action`, planner rollout collection, and planner runtime evaluation surfaces from `projects/snake/snake.{c,h}`.
- Replaced Snake benchmark/training data collection that used oracle/planner actions with deterministic exploration actions that do not inspect path-to-food, action scores, death predictions, or reachable-space outcomes.
- Removed the `snake_planner` benchmark output and planner-dependent Snake tests; `tests/test_shortcut_inventory.c` now rejects reintroducing these Snake oracle/planner surfaces.

## 2026-07-08 Centerline/Breakout oracle/config cleanup

- Removed Centerline oracle action and oracle collection surfaces. Centerline training/benchmark/render paths now use deterministic exploration collection that does not inspect offset to choose a best action.
- Removed Breakout oracle action and oracle token collection surfaces. Breakout training/benchmark/render paths now use deterministic exploration collection that does not inspect ball/paddle geometry to choose a best action.
- Removed `planner_oracle` as an accepted `core/config/layer_config.c` model kind; config tests now assert that the string is rejected.
- Relaxed benchmark tests from oracle-era positive metric thresholds to smoke/contract checks. This avoids treating teacher-data performance as proof of learning, but it also means benchmark quality still needs a legitimate non-oracle improvement path.
- `tests/test_shortcut_inventory.c` now rejects reintroducing Centerline/Breakout oracle symbols and `planner_oracle` config support in active production paths.

## 2026-07-08 decoder and PufferLib pipeline cleanup

- Removed the dead decoder `rollout_score_mode` enum, parser, config field, and tests. With decoder fallback restricted to `none`, keeping rollout score modes was an inert shortcut-shaped knob.
- Removed `pretrained-export`, `checkpoint-export`, and `train-then-export` from `scripts/pufferlib_breakout_pipeline.py`. Those paths loaded or saved PufferLib policy weights to generate data, which violates the current requirement that PufferLib provide environment transitions rather than teacher models.
- Added shortcut inventory coverage so decoder rollout-score config and PufferLib weight-export command surfaces cannot silently return.

## 2026-07-08 mask guardrail cleanup

- Replaced mask config strategies with a single `none` strategy. The old `immediate` and `survival` names exposed guardrail semantics even after Snake masks were neutralized to all-actions-visible behavior.
- Removed all Snake action-mask writer APIs, including the previous all-ones `snake_write_action_mask` compatibility surface. Snake policy inference now selects directly over the full action set.
- Updated config tests to use `strategy = none`; the legacy Snake action-scorer INI was later removed with the action-scorer config cleanup.
- Added shortcut inventory coverage so survival/immediate mask strategies and Snake survival/immediate mask APIs cannot return.

## 2026-07-08 action-scorer config cleanup

- Removed `action_scorer` as a selectable model kind from `core/config/layer_config.{c,h}`.
- Removed action-feature config parsing from layer config. These options were only meaningful for direct action scoring, not next-token policy learning.
- Removed the layer-factory INI constructor for `TkmActionScorer` so configs cannot build direct state/action-feature scorers.
- Deleted the legacy `projects/*/action_scorer.ini` files and replaced the Snake render config with `projects/snake/token_policy.ini`.
- Removed `core/model/action_scorer.c` and `core/train/action_scorer_trainer.c` from the common `CORE_SRC`.
- Deleted the isolated `core/model/action_scorer.{c,h}` and `core/train/action_scorer_trainer.{c,h}` library code and tests. Direct state/action-feature scoring is no longer a maintained model path.
- Added shortcut inventory coverage so active config/factory/render/project paths cannot re-expose `action_scorer` or action-feature config.
- Removed the remaining Snake action-feature constants and writer APIs, including the identity-only rows that had replaced the earlier shortcut features.





## 2026-07-08 direct benchmark-baseline cleanup

- Removed active default benchmark entries that trained directly from continuous or handcrafted features to actions: Centerline trained MLP/MSE, Centerline MLP/CE, Centerline nearest, Centerline linear policy, Breakout nearest, Breakout linear policy, Breakout MLP/CE, Snake nearest multistart, Snake MLP/CE holdout, and Snake MLP DAgger-style holdout.
- The default benchmark now reports tokenized lookup, n-gram, and sparse-lookup paths only. These still need stronger non-oracle performance, but they no longer bypass tokenized observations through direct feature-to-action benchmark baselines.
- Added shortcut inventory coverage so direct feature-to-action benchmark entry points and output names cannot return quietly.

## 2026-07-08 untrained benchmark-policy cleanup

- Removed the untrained `centerline_mlp` benchmark entry. It was a hand-written fixed-weight policy that produced the only positive Centerline benchmark metric without learning from collected data.
- Added shortcut inventory coverage so the exact untrained `tkm_bench_centerline_mlp` entry point and benchmark output name cannot return.

## 2026-07-08 direct model config cleanup

- Removed `mlp_window`, `nearest_policy`, and `linear_policy` as accepted core `[model] kind` values. Runtime layer configs can no longer select those direct feature-to-action policy classes.
- Removed the layer-factory constructor that built `TkmMlpWindow` directly from model INI and parameter files. `core/model/mlp_window.*` remains as a generic MLP backend for next-token models, not as a config-selected action policy path.
- Added shortcut inventory coverage so the direct model config kinds and MLP layer-factory surface cannot return quietly.

## 2026-07-08 generic direct policy library cleanup

- Removed the generic `core/model/linear_policy.*` and `core/model/nearest_policy.*` libraries and their tests. They were direct feature-to-action policy implementations and no remaining supported end-to-end token path needed them.
- Removed their Makefile targets and `CORE_SRC` entries.
- Added shortcut inventory coverage so those generic direct action-policy libraries cannot return quietly.

## 2026-07-08 non-oracle reward learning improvement

- Added return-weighted lookup and sparse-lookup training that choose actions by discounted returns observed in sampled trajectories rather than by hand-coded action scores or environment geometry.
- Added weighted n-gram transitions so next-token action-token prediction can also learn from sampled returns without changing inference into direct action scoring.
- Updated the Centerline lookup, n-gram, and sparse benchmarks to collect state-independent action sweeps from each start and learn from the sampled rewards. This improves benchmark quality without adding oracle labels, death/safety checks, or "best action" functions.
- Updated the Breakout lookup, n-gram, and sparse benchmarks to use richer observation tokens, fixed non-reactive exploration scripts, and sampled-return weighting. The active default Breakout benchmark entries now get paddle hits and score without action-score, intercept-rule, or PufferNet teacher labels.
- Updated Snake lookup and n-gram benchmarks to use richer observation tokens and sampled-return weighting, so the active tokenized paths can collect food without oracle/planner/death-guardrail helpers.
- Added tests that require return-weighted lookup/sparse lookup and weighted n-gram transitions to prefer lower-frequency choices when sampled return is better, and require the Centerline, Breakout, and Snake tokenized benchmark paths to achieve positive task metrics.

## 2026-07-08 masked action/loss surface cleanup

- Removed the remaining public `snake_write_action_mask` API and mask-only tests so Snake no longer exposes a supported action-mask surface at all.
- Removed active Centerline/Breakout masked-CE benchmark variants; those paths used all-ones masks but kept action-mask selection as a supported policy surface.
- Removed the generic `core/head/action_mask` argmax helper and its tests so runtime policy selection cannot silently reintroduce action-validity guardrails.
- Removed `masked_cross_entropy` from train config parsing and the MLP API; active sequence/token training now uses full categorical cross-entropy rather than type/action masks.
- Updated Snake benchmark MLP inference and DAgger-style rollouts to use plain categorical argmax over all actions.
- Added shortcut inventory coverage so masked action/loss surfaces cannot return through Makefile, benchmark, train config, MLP, or tests.

## 2026-07-08 PufferLib typed next-token cleanup

- Removed `typed_next_token` and `typed_next_token_unmasked` action-head modes from PufferLib token configs and policy parsing. Token-stream models now use the categorical full-vocabulary head.
- Removed token-MLP action/observation output masks and typed argmax; training uses full-vocabulary cross-entropy, and runtime converts only action-class predictions to actions.
- Replaced n-gram separate observation/action target tables with one full-vocabulary class table keyed only by token-stream context.
- Added shortcut inventory coverage so typed next-token heads, target-type hashing, masked token-head training, and split n-gram action/observation count tables cannot return.

## 2026-07-08 PufferLib feature-selection cleanup

- Removed action-label-based observation dimension selection from `projects/breakout/pufferlib_policy.{c,h}`. The old `action_separation` and `first_then_action_separation` modes selected dimensions by how strongly they separated recorded actions.
- Changed PufferLib token configs that used `action_separation` to fixed `first_dims` selection and deleted the first-then-action config from the benchmark matrix.
- Removed per-action observation statistics and best-score dimension picking from token-policy training.
- Added shortcut inventory coverage so action-label-based PufferLib observation selection cannot return.

## 2026-07-08 PufferLib direct action token-mode cleanup

- Removed `predict = action` PufferLib token configs and deleted the direct action-token config files for `mlp_window`, `linear_policy`, `action_mlp_from_token_context`, and delta-action-history variants.
- Removed PufferLib token-policy parser/training/inference support for direct observation-to-action MLP/linear models and action-only MLP-from-token-context models. Supported PufferLib token policies now use next-token token-stream models: `token_ngram`, `token_backoff_ngram`, and `token_mlp_window`.
- Removed those direct action-token configs from the matrix benchmark and changed PufferLib app defaults to `projects/breakout/pufferlib_token_ngram.ini` so smoke, benchmark, and render entry points do not point at deleted direct-action configs.
- Added shortcut inventory coverage so the direct modes, config files, stale app defaults, and benchmark labels cannot return quietly.
