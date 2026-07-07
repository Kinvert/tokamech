# Head Layer Options

Date: 2026-07-06

## Current implementation status

The first categorical head module is implemented in core:

```text
core/head/categorical.h
core/head/categorical.c
tests/test_categorical_head.c
```

Implemented functions:

```text
tkm_categorical_softmax
tkm_categorical_argmax
```

This is inference math only. It does not train model weights.

## What the head layer does

Sequence model:

```text
context embedding -> hidden/logit vector
```

Head:

```text
hidden/logit vector -> probability distribution or selected token
```

Example:

```text
logits = [1.0, 2.0, 3.0]
softmax -> probabilities
argmax -> token/action index
```

## Current head option

Categorical:

```ini
[head]
kind = categorical
```

Good for:

```text
next-token prediction
discrete action prediction
VQ code prediction
packed token prediction
classification-style auxiliary heads
```

The implementation uses max-subtracted softmax so large logits do not overflow.

## Current tests

`tests/test_categorical_head.c` covers:

```text
softmax probabilities sum to 1
larger logits produce larger probabilities
large-logit numerical stability
argmax returns first max index
bad input handling
```

## Planned head options

Useful future heads:

```text
multi_categorical         several independent categorical outputs
binary                    terminal/collision/success prediction
scalar_regression         value, distance, reward, continuous scalar
gaussian                  mean/std continuous action head
mixture_gaussian          multimodal continuous actions
vq_code_prediction        categorical over VQ codes
value                     RL value function head
reward_prediction         auxiliary reward model head
terminal_prediction       auxiliary terminal model head
```

## Architecture rule

Heads should not own project semantics.

Good:

```text
float logits[action_count] -> uint16_t action_token
```

Bad:

```text
if project == line_follower then clamp motor left/right here
```

Project-specific action rules belong in decoders and safety filters. Heads only convert model outputs into generic prediction forms.
