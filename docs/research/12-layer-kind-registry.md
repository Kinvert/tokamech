# Layer Kind Registry

Date: 2026-07-06

## Current implementation status

The first layer-kind registry is implemented in core:

```text
core/config/layer_config.h
core/config/layer_config.c
tests/test_layer_config.c
```

Implemented functions:

```text
tkm_tokenizer_kind_from_string
tkm_embedder_kind_from_string
tkm_sequence_layout_kind_from_string
tkm_model_kind_from_string
tkm_head_kind_from_string
tkm_decoder_kind_from_string
tkm_loss_kind_from_string
tkm_layer_config_from_ini
```

This maps INI strings to enums. It does not construct layer objects yet.

## Current defaults

An empty config produces conservative phase-0 defaults:

```ini
[tokenizer]
kind = int_bins

[embedder]
kind = none

[sequence_layout]
kind = obs_action_interleaved

[model]
kind = lookup_policy

[head]
kind = categorical

[decoder]
kind = argmax

[loss]
kind = cross_entropy
```

These defaults match the current simplest working stack.

## Recognized tokenizer kinds

```text
int_bins
packed_token
raw_continuous
vq_code
normalized_continuous
```

Implemented as reusable core modules:

```text
int_bins
vq_code
raw_continuous by continuous vectorizer raw mode
normalized_continuous by continuous vectorizer standardize mode
```

Project-specific or conceptual:

```text
packed_token
```

`packed_token` currently exists as project code in Snake, not as a generic core packer.

## Recognized embedder kinds

```text
none
lookup
linear
```

Implemented:

```text
none by convention
lookup
linear
```

## Recognized sequence layout kinds

```text
obs_action_interleaved
joint_transition
```

Implemented:

```text
obs_action_interleaved
joint_transition
```

## Recognized model kinds

```text
lookup_policy
planner_oracle
ngram
mlp_window
sparse_lookup
nearest_policy
linear_policy
action_scorer
transformer_decoder
```

Implemented:

```text
lookup_policy
planner_oracle for Snake only
ngram
mlp_window
sparse_lookup
nearest_policy
linear_policy
action_scorer
```

Recognized but not implemented as generic core models yet:

```text
transformer_decoder
```

The registry includes these names so config files can stabilize before the models exist.

## Recognized head kinds

```text
categorical
multi_categorical
scalar_regression
```

Implemented:

```text
categorical
```

Recognized but not implemented as generic core heads yet:

```text
multi_categorical
scalar_regression
```

## Recognized decoder kinds

```text
argmax
sample
inverse_bins
discrete_action
```

Implemented:

```text
argmax by categorical helper usage
discrete_action
```

Recognized but not implemented as generic core decoders yet:

```text
sample
inverse_bins as a standalone generic decoder
```

## Recognized loss kinds

```text
cross_entropy
mse
huber
binary_cross_entropy
weighted_sum
action_smoothness
```

Implemented as core math:

```text
cross_entropy
mse
huber
binary_cross_entropy
weighted_sum
action_smoothness
```

## Current tests

`tests/test_layer_config.c` covers:

```text
read known kinds from INI
phase-0 defaults
string parser success cases
unknown string rejection
unknown INI kind rejection
```

## Architecture rule

The registry is allowed to know names and enums.

It should not own project construction logic.

Good:

```text
"vq_code" -> TKM_TOKENIZER_VQ_CODE
```

Bad:

```text
if project == snake and tokenizer == vq_code then load this hardcoded codebook
```

The next layer should be a project-aware factory or setup function that reads project-specific config and constructs selected core modules.
