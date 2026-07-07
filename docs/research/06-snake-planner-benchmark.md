# Snake Planner Benchmark Findings

Date: 2026-07-06

## What failed

The first Snake renderer worked, but the policy looked bad because the tested success condition was too weak.

The old lookup-policy test only proved this:

```text
eat one food item, then stop the rollout
```

That is not the same as playing Snake. It hides the real quality problem.

## Baseline benchmark before the planner option

```text
centerline steps=8 score=1 metric=1
breakout steps=220 score=2 metric=3
snake_lookup steps=3 score=1 metric=1
```

`snake_lookup` collected one food in three steps because the rollout intentionally stopped after the first food. It did not prove survival, route quality, or repeated collection.

## Root cause

The compact Snake token is useful for a small lookup baseline, but it is lossy:

```text
token = direction + relative food direction + local danger bits
```

That token does not preserve full body geometry. A lookup/count policy trained on short oracle snippets can choose the common action for a local situation, but it cannot reliably reason about future self-collisions or traps.

The first BFS oracle also had a failure mode: it greedily chased food and could enter a small pocket after eating. This collected several foods quickly, then died when the snake boxed itself in.

## Added layer option

Snake now has a `planner_policy` option:

```ini
[model]
kind = planner_oracle

[planner]
kind = safe_bfs
```

This is not a learned token model. It is a deterministic control baseline. Its purpose is to prove the environment is solvable, provide benchmark numbers, and give the renderer a competent default while learned policies catch up.

The planner chooses actions in this order:

```text
1. Try the BFS action toward food.
2. Simulate the candidate move.
3. Reject it if the next state terminals or leaves too little reachable space.
4. If rejected, score all legal actions by reachable space, food reward, and food distance.
5. Fall back to the first safe action only if no scored action is available.
```

This keeps the code easy to audit: no neural network, no hidden state, no magic constants beyond a minimum-space guard.

## Benchmark after the planner option

```text
centerline steps=8 score=1 metric=1
breakout steps=220 score=2 metric=3
snake_lookup steps=3 score=1 metric=1
snake_sparse_lookup steps=96 score=11 metric=11
snake_planner steps=96 score=11 metric=11
```

Interpretation:

```text
centerline unchanged
breakout unchanged
lookup Snake remains a weak token baseline
full-state sparse lookup Snake matches the deterministic planner on the benchmark rollout
planner Snake collects 11 foods over the 96-step benchmark without terminal
```

## Architecture lesson

This supports the interchangeable-layer architecture.

Different projects need different policy layers at different maturity levels:

```text
lookup_policy      simple token sanity check
sparse_lookup      exact-token imitation for large or hashed state tokens
planner_oracle     deterministic benchmark ceiling
mlp_window         first learned continuous/discrete model
transformer        next-token sequence model
vq_code            learned discrete representation layer
```

For Snake, `planner_oracle` is currently the right renderer default because it makes the environment visibly playable. `sparse_lookup` now proves that planner rollouts can be distilled into an exact-token policy for the deterministic benchmark. `lookup_policy` should remain in the benchmark because it is useful evidence: compact tokens and dense count policies are not enough for this task.

## Next useful experiments

Good next layer experiments:

```text
sparse_lookup_more_starts: distill planner rollouts from multiple deterministic starts
packed_token_v2: include richer body/food geometry
continuous_features: normalized head, food, and danger distances
mlp_window: predict action from recent continuous observations
vq_code: learn discrete state codes from continuous Snake observations
transformer_decoder: predict actions from observation/action history
```

The benchmark rule should stay strict: every new method must run across centerline, breakout, and Snake without regressing existing metrics.
