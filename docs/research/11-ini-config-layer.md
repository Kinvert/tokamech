# INI Config Layer

Date: 2026-07-06

## Current implementation status

The first config parser is implemented in core:

```text
core/config/ini.h
core/config/ini.c
tests/test_ini.c
```

Implemented functions:

```text
tkm_ini_parse
tkm_ini_get
tkm_ini_get_i32
tkm_ini_get_f32
```

This parses INI text from memory. File loading and layer object creation are not implemented yet.

## Supported syntax

Sections:

```ini
[tokenizer]
kind = vq_code
```

Key/value pairs:

```ini
dim = 4
delta = 1.0
```

Comments:

```ini
# comment
; comment
```

Later entries override earlier entries for lookup:

```ini
[model]
kind = lookup_policy
kind = planner_oracle
```

`tkm_ini_get(&ini, "model", "kind", "")` returns:

```text
planner_oracle
```

## Why this matters

The architecture goal is that projects can switch layer methods from an `.ini` file:

```ini
[tokenizer]
kind = int_bins

[embedder]
kind = lookup

[sequence_layout]
kind = obs_action_interleaved

[model]
kind = lookup_policy

[loss]
kind = cross_entropy
```

The parser is the first step. It does not instantiate those layers yet.

## Current tests

`tests/test_ini.c` covers:

```text
sections
key/value parsing
comments
whitespace
string lookup
int lookup
float lookup
default values
duplicate override behavior
bad input handling
```

## Intended next layer

The next config layer should map strings to enums:

```text
"int_bins" -> TKM_TOKENIZER_INT_BINS
"vq_code" -> TKM_TOKENIZER_VQ_CODE
"lookup" -> TKM_EMBEDDER_LOOKUP
"linear" -> TKM_EMBEDDER_LINEAR
"obs_action_interleaved" -> TKM_SEQUENCE_LAYOUT_OBS_ACTION_INTERLEAVED
"cross_entropy" -> TKM_LOSS_CROSS_ENTROPY
```

After that, project code can do:

```text
read project ini
parse layer kinds
construct selected tokenizer/embedder/layout/model/loss
run benchmark
```

## Architecture rule

The INI parser should not know about line followers, Snake, Breakout, or any specific paper method.

Good:

```text
section + key -> string/int/float value
```

Bad:

```text
if tokenizer.kind == vq_code then initialize Snake codebook
```

Layer construction belongs in a config/registry layer above the parser. Project-specific values belong in the project folder.
