# Agent Notes

## Long-running goals

Do not launch GUI or display commands during long-running or unattended goals unless the user explicitly asks for them. Do not block yourself on a render window while the user may be away.

This includes commands like:

```sh
DISPLAY=:0 build/snake_render
DISPLAY=:0 make snake-render
DISPLAY=:0 timeout 4s build/snake_render
```

Headless checks are fine:

```sh
make test
make benchmark
make build/snake_render
```

For RayLib or other visual checks, build the binary headlessly first. If a visual run would be useful, stop and ask the user before using `DISPLAY=:0`.
