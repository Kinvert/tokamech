# Snake Safety Filter

Date: 2026-07-06

## Why this was added

Nearest-neighbor imitation over continuous Snake features matches the trained multistart benchmark:

```text
snake_nearest_multistart steps=288 score=34 metric=34
```

But a probe on held-out first-food positions showed pure nearest-neighbor actions can terminal early. That is useful evidence: continuous nearest-neighbor matching is still not enough to trust as an actuator.

## Safety filter behavior

Snake now exposes:

```text
snake_filter_unsafe_action
```

It does one narrow thing:

```text
simulate the suggested action for one step
if it does not terminal, keep it
otherwise try the fallback action
if fallback also terminals, use the first safe action
```

The safety filter does not search for long-term optimality. It only rejects actions that immediately crash.

## Held-out benchmark

Training starts:

```text
{6, 8}
{3, 5}
{8, 5}
```

Held-out evaluation starts:

```text
{6, 2}
{3, 8}
{8, 8}
{1, 1}
{1, 8}
```

Command:

```sh
make benchmark
```

Output line:

```text
snake_nearest_safe_holdout steps=480 score=47 metric=47
```

## What this proves

This proves:

```text
learned action suggestion and safety filtering can be composed cleanly
Snake can survive held-out deterministic starts with nearest-neighbor suggestions plus immediate safety checks
the benchmark can expose pure-policy failures without allowing unsafe actions through
```

## What this does not prove

This does not prove:

```text
pure nearest-neighbor policy generalizes
the safety filter is enough for robotics
long-horizon safety is solved
neural policy training works
```

For robotics, this is the right architectural direction: the learned policy suggests actions, but project-specific safety code gets the final veto before actuation. For this repo, the next target should be reducing safety interventions by improving the learned policy layer.
