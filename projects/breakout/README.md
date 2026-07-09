# Breakout

`projects/breakout` contains Tokamech Breakout experiments and the Tokamech-side PufferLib Breakout policy bridge.

It keeps the important PufferLib shape:

```text
screen: 576 x 330
bricks: 18 columns x 6 rows
actions: NOOP, LEFT, RIGHT
observations: 10 scalar features + 108 brick states = 118 floats
renderer: RayLib
```

The original standalone proof uses Tokamech's local token path:

```text
flat Breakout observations
-> compact relative ball/paddle token
-> deterministic exploration token trajectories
-> lookup/count policy
-> closed-loop runtime
```

The current PufferLib bridge uses literal PufferLib Breakout JSONL as the source of truth:

```text
PufferLib Breakout JSONL
-> selected quantized observation tokens
-> autoregressive obs/action token stream
-> config-selected next-token sequence model
-> categorical full-vocabulary next-token head
-> PufferLib Breakout c_step
```

Config-selected next-token stacks:

```text
projects/breakout/pufferlib_token_ngram.ini
projects/breakout/pufferlib_token_backoff_ngram.ini
projects/breakout/pufferlib_token_mlp_first_dims.ini
projects/breakout/pufferlib_token_mlp_window.ini
projects/breakout/pufferlib_token_mlp_window_unmasked.ini
```

PufferLib-side requirements for this bridge are documented in:

```text
projects/breakout/PUFFERLIB_INTEGRATION.md
```

RayLib render overlays are selected in config with:

```ini
[render]
token_visualization = token_stream
```

Accepted values are `none`, `topk`, `token_stream`, `continuous_mlp`,
`context_grid`, `quantization`, `pipeline`, `transformer`, and `all`.

Headless PufferLib benchmark:

```sh
make build/pufferlib_breakout_benchmark
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_ngram.ini \
./build/pufferlib_breakout_benchmark
```

Run the no-render config matrix against a JSONL dataset:

```sh
make pufferlib-breakout-matrix-smoke \
  TKM_PUFFERLIB_BREAKOUT_JSONL=/tmp/tkm_breakout_train_full_export_100k_v2/transitions-000000.jsonl
```

Run headless tests:

```sh
make test
```

Run the RayLib demo:

```sh
DISPLAY=:0 make breakout-render
```

Controls:

```text
Hold LEFT_SHIFT for human control
A / Left Arrow = left
D / Right Arrow = right
R = reset
ESC = close
```
