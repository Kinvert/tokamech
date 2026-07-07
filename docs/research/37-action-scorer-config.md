# Action Scorer Config Support

Date: 2026-07-06

## What changed

The action scorer is now recognized by the layer registry:

```ini
[model]
kind = action_scorer
```

It can also be constructed by the layer factory:

```text
tkm_layer_factory_init_action_scorer
```

## INI shape

```ini
[model]
kind = action_scorer
state_dim = 1
action_dim = 2
hidden_dim = 1
w0 = scorer_w0
b0 = scorer_b0
w1 = scorer_w1
b1 = scorer_b1
```

Flat params:

```text
scorer_w0 = 0.0 1.0 1.0
scorer_b0 = 0.0
scorer_w1 = 1.0
scorer_b1 = 0.0
```

## Why this matters

Before this, `core/model/action_scorer` worked as C code and benchmarks could call it directly, but a project config could not select it as a model kind.

This moves the scorer closer to the repo's intended architecture:

```text
INI chooses method
factory constructs core layer
project code writes explicit state/action features
runtime or benchmark calls the selected layer
```

## Current limits

This is still scalar/flat-param construction only.

It does not yet build a whole project runtime from one INI file, and it does not move DAgger-style training into core. Those are separate follow-up steps.

## Tests

Coverage was added in:

```text
tests/test_layer_config.c
tests/test_layer_factory.c
```

The tests verify:

```text
action_scorer string maps to TKM_MODEL_ACTION_SCORER
factory builds TkmActionScorer from INI plus params
factory rejects wrong-shaped scorer params
```
