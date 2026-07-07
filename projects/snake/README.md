# Snake

`projects/snake` is Tokamech Phase 2: a deterministic CPU-only grid Snake environment for testing token policies.

It is inspired by simple PufferLib/Ocean-style C envs:

```text
flat observations
discrete actions
rewards
terminals
grid/body state
RayLib renderer
```

Phase 2 uses the current Tokamech token path:

```text
flat Snake env state
-> compact relative-food/local-danger/direction token
-> BFS oracle trajectories
-> lookup/count policy
-> closed-loop runtime
```

Run tests:

```sh
make test
```

Run the RayLib renderer:

```sh
DISPLAY=:0 make snake-render
```

Controls:

```text
Hold LEFT_SHIFT for human control
Arrow keys or WASD move
R resets
ESC closes
```
