# PufferLib Breakout Autoregressive Token Stream

Date: 2026-07-07.

## Goal

Move the PufferLib Breakout bridge closer to "Humanoid Locomotion as Next Token Prediction" by adding a true typed next-token path. This is v003. It does not replace the v002 `mlp_window` or `linear_policy` paths.

The important distinction:

```text
v002: observation/action history -> action token
v003: typed observation/action stream -> next typed token
```

During training, v003 predicts both observation tokens and action tokens. During real PufferLib evaluation, only predicted action tokens are executed. Observation tokens in the runtime context come from real `env.observations` after `c_step`, not from model imagination.

## Config

The first v003 config is:

```text
projects/breakout/pufferlib_token_ngram.ini
```

Layer shape:

```ini
[sequence_layout]
kind = autoregressive_obs_action
history = 8
predict = next_token

[tokenizer.observation]
kind = selected_quantized
dims = 8
selection = action_separation
bins = 16

[tokenizer.action]
kind = categorical
actions = 3

[sequence_model]
kind = token_ngram
epochs = 2
learning_rate = 0.10

[action_head]
kind = typed_next_token
actions = 3
```

The model is intentionally small: a compact hashed n-gram table over typed stream context. For observation outputs it keeps a few candidate tokens per context. For actions it keeps action counts and a learned action-prior fallback when the exact runtime context is unseen.

## Training Stream

For each dataset row, the selected observation dimensions are quantized and emitted as typed OBS tokens, then the action is emitted as an ACTION token:

```text
BOS
OBS(dim0 token)
OBS(dim1 token)
...
ACTION(action)
OBS(dim0 token)
...
ACTION(action)
```

The held-out pass reports:

```text
obs_token_accuracy
action_token_accuracy
heldout_accuracy
```

`heldout_accuracy` is kept for compatibility and equals action-token accuracy for this backend.

## Runtime

The render and benchmark adapters still own the real PufferLib loop:

```text
read PufferLib env.observations
insert real OBS tokens into Tokamech stream context
predict next ACTION token
write env.actions[0]
call PufferLib c_step
repeat with real next env.observations
```

This keeps the PufferLib environment authoritative. Predicted observation tokens are a training target, not the world state used for control.

## Smoke Result

Command:

```sh
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_ngram.ini \
TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES=1 \
TKM_PUFFERLIB_BREAKOUT_MAX_STEPS=8000 \
TKM_PUFFERLIB_BREAKOUT_EVAL_SEED=0 \
./build/pufferlib_breakout_benchmark \
  /tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl \
  /tmp/tkm_v003_token_ngram_smoke
```

Observed:

```text
token_layers layout=autoregressive_obs_action tokenizer=selected_quantized model=token_ngram head=typed_next_token
token_train rows=100000 train_rows=80000 heldout_rows=20000 heldout_accuracy=0.393 obs_token_accuracy=0.047 action_token_accuracy=0.393 bins=16 dims=8 history=8 epochs=2
method=token_ngram_all rows=100000 episodes=1 mean_return=7.000 max_return=7.000 mean_length=363.000 steps=363 actions=[41,15,307]
```

This is much weaker than `mlp_window`. The result is still useful because it proves the new obs/action next-token contract can train from the same JSONL and run through the same real PufferLib adapter.

## Current Limits

This is not yet a transformer or humanoid-scale model. The current n-gram backend has short context, sparse generalization, and low observation-token accuracy on Breakout. The next useful layer swap is a stronger `sequence_model.kind` behind the same `autoregressive_obs_action` layout, such as a small recurrent model or transformer-style decoder.
