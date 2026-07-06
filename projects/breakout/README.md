# Breakout

`projects/breakout` is a Tokamech token-policy clone of PufferLib's `ocean/breakout` environment.

It keeps the important PufferLib shape:

```text
screen: 576 x 330
bricks: 18 columns x 6 rows
actions: NOOP, LEFT, RIGHT
observations: 10 scalar features + 108 brick states = 118 floats
renderer: RayLib
```

Instead of RL/PufferNet, Phase 1 uses Tokamech's current token path:

```text
flat Breakout observations
-> compact relative ball/paddle token
-> oracle token trajectories
-> lookup/count policy
-> closed-loop runtime
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
