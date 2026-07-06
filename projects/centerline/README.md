# Centerline

`centerline` is the Phase 0 deterministic test project for Tokamech.

It is a tiny control problem:

```text
state: signed offset from zero
observation: obs[0] = offset
actions: left, stay, right
goal: drive offset to zero
```

The oracle policy is intentionally obvious:

```text
offset < 0 -> right
offset > 0 -> left
offset == 0 -> stay
```

This project exists to prove the CPU-only end-to-end stack before adding line-follower physics or CUDA.

Run:

```sh
make test
```

Optional RayLib render demo:

```sh
make centerline-render
```

The render target is intentionally separate from `make test` so the deterministic test path stays headless. The Makefile looks for `raylib-5.5_linux_amd64` in this repo first and falls back to `/home/claude/pathfinder/raylib-5.5_linux_amd64` in this workspace.

To run on the local display:

```sh
DISPLAY=:0 make centerline-render
```

If RayLib is somewhere else, pass the directory explicitly:

```sh
make centerline-render RAYLIB_DIR="/path/to/raylib-5.5_linux_amd64"
```
