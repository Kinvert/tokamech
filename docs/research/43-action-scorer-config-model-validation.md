# Action Scorer Config Model Validation

Date: 2026-07-06

## What changed

The config-driven Snake action-scorer benchmark now validates:

```ini
[model]
kind = action_scorer
```

before using action-scorer training.

It also still validates:

```ini
[train]
loss = margin
```

or:

```ini
[train]
loss = masked_cross_entropy
```

## Why this matters

Before this change, the benchmark used `[train] loss` but ignored `[model] kind`.

That was misleading. A config like this:

```ini
[model]
kind = lookup_policy

[train]
loss = masked_cross_entropy
```

would still run the action-scorer benchmark.

Now it fails cleanly. The config file must agree with the benchmark path.

## Result

The valid project-local config still works:

```text
snake_action_scorer_config_file_holdout steps=480 score=58 metric=58
```

Wrong model kinds return an empty benchmark result and are covered by tests.

## Current limit

This is validation inside the Snake action-scorer benchmark, not a general project runner. The next step is still a common runner that selects the model implementation from `[model] kind` instead of requiring a model-specific benchmark entry point.
