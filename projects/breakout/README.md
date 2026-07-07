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
-> oracle token trajectories
-> lookup/count policy
-> closed-loop runtime
```

The current PufferLib bridge uses literal PufferLib Breakout JSONL as the source of truth:

```text
PufferLib Breakout JSONL
-> selected continuous observation tokens
-> obs/action history
-> config-selected sequence model
-> categorical action token
-> PufferLib Breakout c_step
```

Config-selected token stacks:

```text
projects/breakout/pufferlib_token_mlp.ini
projects/breakout/pufferlib_token_linear.ini
projects/breakout/pufferlib_token_ngram.ini
```

Headless PufferLib benchmark:

```sh
make build/pufferlib_breakout_benchmark
TKM_PUFFERLIB_BREAKOUT_CONFIG=projects/breakout/pufferlib_token_mlp.ini \
./build/pufferlib_breakout_benchmark
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
