# PufferLib Breakout integration requirements

Tokamech treats PufferLib as an environment transition source, not as a teacher
policy. The PufferLib side must expose Breakout state transitions and dataset
export hooks while leaving all tokenization, training, and inference inside
Tokamech.

## Required PufferLib files and symbols

Tokamech builds the PufferLib Breakout bridge with `PUFFERLIB_DIR` pointing at
the PufferLib checkout. The following files must exist under that checkout:

```text
ocean/breakout/breakout.h
ocean/breakout/binding.c
ocean/breakout/exporter.h
resources/shared/puffers_128.png
```

The Breakout C API must provide the same symbols used by Tokamech:

```text
Breakout
StaticVec
allocate
free_allocated
c_reset
c_step
breakout_vec_step_range
breakout_export_close
```

`breakout.h` must expose the Breakout environment type and single-env stepping
API used by the smoke, benchmark, and render binaries. `binding.c` must expose
the vector stepping path used by the train-export hook test.

## Environment contract

The Breakout environment shape must stay stable:

```text
screen: 576 x 330
bricks: 18 columns x 6 rows
actions: 0 = NOOP, 1 = LEFT, 2 = RIGHT
observations: 10 scalar features + 108 brick states = 118 floats
```

Tokamech expects discrete actions. Continuous-action-only Breakout builds are
not sufficient for this bridge.

## Export hook contract

PufferLib Breakout must support headless transition export controlled by these
environment variables:

```text
PUFFERLIB_BREAKOUT_EXPORT_JSONL
PUFFERLIB_BREAKOUT_EXPORT_MANIFEST
PUFFERLIB_BREAKOUT_EXPORT_SOURCE
PUFFERLIB_BREAKOUT_EXPORT_MAX_ROWS
PUFFERLIB_BREAKOUT_DEMO_EXPORT_STEPS
```

When `PUFFERLIB_BREAKOUT_EXPORT_JSONL` is set, `breakout_vec_step_range` should
append one JSONL row per stepped agent until `PUFFERLIB_BREAKOUT_EXPORT_MAX_ROWS`
is reached. `breakout_export_close` must flush and close the export files and
write the manifest.

Each JSONL row must contain:

```json
{
  "version": 1,
  "env": "pufferlib_breakout",
  "source": "train_export",
  "env_index": 0,
  "episode": 0,
  "t": 0,
  "obs_dim": 118,
  "obs": [],
  "action": 0,
  "reward": 0.0,
  "terminal": 0
}
```

The `obs` array must contain exactly 118 floats captured from the environment
state associated with that transition. `action` must be the discrete action
executed for that row. `reward` and `terminal` must be the post-step result.

The manifest must include at least:

```json
{
  "env": "pufferlib_breakout",
  "source": "train_export",
  "obs_dim": 118,
  "action_count": 3,
  "row_count": 0
}
```

Tokamech accepts additional manifest fields, but these fields are required.

## Standalone demo export

`scripts/pufferlib_breakout_pipeline.py standalone-demo-export` expects a
headless PufferLib Breakout binary at:

```text
$PUFFERLIB_DIR/breakout
```

That binary must honor the export variables above plus
`PUFFERLIB_BREAKOUT_DEMO_EXPORT_STEPS`. It should run without opening a render
window when the export variables are set.

## What PufferLib must not provide

Do not add PufferLib-side shortcuts for Tokamech:

```text
pretrained policy export
checkpoint policy export
PufferNet weight export
intercept-rule labels
oracle action labels
best-action labels
death/safety masks
direct observation-to-action teacher predictions
```

Tokamech should receive environment transitions only. It then loads JSONL,
quantizes observations into tokens, trains next-token models, and performs
closed-loop inference from those token models.

## Tokamech verification commands

After changing PufferLib, run these from the Tokamech checkout:

```sh
make build/pufferlib_breakout_smoke
make build/pufferlib_breakout_benchmark
make build/pufferlib_breakout_policy_render
make build/test_pufferlib_breakout_train_export_hook
./build/test_pufferlib_breakout_train_export_hook
```

For a full Tokamech check:

```sh
make test
make benchmark
```

## Pathfinder branch diff

This is the full current diff for the PufferLib-side branch/worktree used by
Tokamech:

```text
branch: tokamech-breakout-jsonl
base: prophop
includes: committed export hooks plus the working-tree c_render overlay and side-panel hook
ocean/breakout/binding.c                      |  43 +++
 ocean/breakout/breakout.c                     |  54 +++-
 ocean/breakout/breakout.h                     |  20 +-
 ocean/breakout/exporter.h                     | 376 ++++++++++++++++++++++++++
 ocean/breakout/tests/run_exporter_smoke.sh    |  24 ++
 ocean/breakout/tests/test_breakout_exporter.c | 125 +++++++++
 ocean/craftax/binding.c                       |   3 +
 src/pufferlib.cu                              |  12 +-
 src/vecenv.h                                  |  19 +-
 9 files changed, 660 insertions(+), 16 deletions(-)
```

Untracked Pathfinder files such as the local `breakout` binary and
`screenrec001.gif` are not part of this diff.

```diff
diff --git a/ocean/breakout/binding.c b/ocean/breakout/binding.c
index 471e78b9..5b7ff794 100644
--- a/ocean/breakout/binding.c
+++ b/ocean/breakout/binding.c
@@ -1,12 +1,55 @@
 #include "breakout.h"
+#include "exporter.h"
 #define OBS_SIZE 118
 #define NUM_ATNS 1
 #define ACT_SIZES {3}
 #define OBS_TENSOR_T FloatTensor

+struct StaticVec;
+void breakout_vec_step(struct StaticVec* vec);
+void breakout_vec_step_range(struct StaticVec* vec, int env_start, int env_count, int num_workers);
+
+#define MY_VEC_STEP breakout_vec_step
+#define MY_VEC_STEP_RANGE breakout_vec_step_range
 #define Env Breakout
 #include "vecenv.h"

+void breakout_vec_step(StaticVec* vec) {
+    memset(vec->rewards, 0, vec->total_agents * sizeof(float));
+    memset(vec->terminals, 0, vec->total_agents * sizeof(float));
+
+    Breakout* envs = (Breakout*)vec->envs;
+    if (!breakout_export_enabled()) {
+        #pragma omp parallel for schedule(static)
+        for (int i = 0; i < vec->size; i++) {
+            c_step(&envs[i]);
+        }
+        return;
+    }
+
+    #pragma omp parallel for schedule(static)
+    for (int i = 0; i < vec->size; i++) {
+        breakout_export_step_env(&envs[i], i);
+    }
+}
+
+void breakout_vec_step_range(StaticVec* vec, int env_start, int env_count, int num_workers) {
+    Breakout* envs = (Breakout*)vec->envs;
+    int env_end = env_start + env_count;
+    if (!breakout_export_enabled()) {
+        #pragma omp parallel for schedule(static) num_threads(num_workers)
+        for (int i = env_start; i < env_end; i++) {
+            c_step(&envs[i]);
+        }
+        return;
+    }
+
+    #pragma omp parallel for schedule(static) num_threads(num_workers)
+    for (int i = env_start; i < env_end; i++) {
+        breakout_export_step_env(&envs[i], i);
+    }
+}
+
 void my_init(Env* env, Dict* kwargs) {
     env->num_agents = 1;
     env->frameskip = dict_get(kwargs, "frameskip")->value;
diff --git a/ocean/breakout/breakout.c b/ocean/breakout/breakout.c
index 84029740..10496a8e 100644
--- a/ocean/breakout/breakout.c
+++ b/ocean/breakout/breakout.c
@@ -1,13 +1,11 @@
 #include <time.h>
 #include "breakout.h"
+#include "exporter.h"
 #include "puffernet.h"

-void demo() {
-    Weights* weights = load_weights("resources/breakout/breakout_weights.bin");
-    int logit_sizes[1] = {3};
-    PufferNet* net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);
-
-    Breakout env = {
+static void configure_demo_env(Breakout* env) {
+    *env = (Breakout){
+        .num_agents = 1,
         .frameskip = 1,
         .width = 576,
         .height = 330,
@@ -25,6 +23,42 @@ void demo() {
         .paddle_speed = 620,
         .continuous = 0,
     };
+}
+
+static PufferNet* load_demo_policy(Weights** weights_out) {
+    Weights* weights = load_weights("resources/breakout/breakout_weights.bin");
+    int logit_sizes[1] = {3};
+    PufferNet* net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);
+    *weights_out = weights;
+    return net;
+}
+
+void demo_export(int steps) {
+    setenv("PUFFERLIB_BREAKOUT_EXPORT_SOURCE", "demo_export", 0);
+
+    Weights* weights = NULL;
+    PufferNet* net = load_demo_policy(&weights);
+    Breakout env;
+    configure_demo_env(&env);
+    allocate(&env);
+    c_reset(&env);
+
+    for (int step = 0; step < steps; step++) {
+        forward_puffernet(net, env.observations, env.actions);
+        breakout_export_step_env(&env, 0);
+    }
+
+    breakout_export_close();
+    free_puffernet(net);
+    free(weights);
+    free_allocated(&env);
+}
+
+void demo() {
+    Weights* weights = NULL;
+    PufferNet* net = load_demo_policy(&weights);
+    Breakout env;
+    configure_demo_env(&env);
     allocate(&env);

     env.client = make_client(&env);
@@ -60,5 +94,13 @@ void demo() {
 }

 int main() {
+    const char* demo_steps = getenv("PUFFERLIB_BREAKOUT_DEMO_EXPORT_STEPS");
+    if (demo_steps != NULL && demo_steps[0] != '\0') {
+        int steps = atoi(demo_steps);
+        if (steps > 0) {
+            demo_export(steps);
+            return 0;
+        }
+    }
     demo();
 }
diff --git a/ocean/breakout/breakout.h b/ocean/breakout/breakout.h
index a4a1dad4..9ca2aa6d 100644
--- a/ocean/breakout/breakout.h
+++ b/ocean/breakout/breakout.h
@@ -508,6 +508,12 @@ void c_step(Breakout* env) {

 Color BRICK_COLORS[6] = {RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE};

+typedef void (*BreakoutRenderOverlayCallback)(Breakout* env, void* user);
+
+static BreakoutRenderOverlayCallback g_breakout_render_overlay = NULL;
+static void* g_breakout_render_overlay_user = NULL;
+static int g_breakout_render_overlay_panel_width = 0;
+
 Client* make_client(Breakout* env) {
     Client* client = (Client*)calloc(1, sizeof(Client));
     client->width = env->width;
@@ -517,7 +523,7 @@ Client* make_client(Breakout* env) {
     client->ball_width = env->ball_width;
     client->ball_height = env->ball_height;

-    InitWindow(env->width, env->height, "PufferLib Breakout");
+    InitWindow(env->width + g_breakout_render_overlay_panel_width, env->height, "PufferLib Breakout");
     SetTargetFPS(60 / env->frameskip);

     client->ball = LoadTexture("resources/shared/puffers_128.png");
@@ -529,6 +535,15 @@ void close_client(Client* client) {
     free(client);
 }

+void breakout_set_render_overlay(BreakoutRenderOverlayCallback callback, void* user) {
+    g_breakout_render_overlay = callback;
+    g_breakout_render_overlay_user = user;
+}
+
+void breakout_set_render_overlay_panel_width(int width) {
+    g_breakout_render_overlay_panel_width = width > 0 ? width : 0;
+}
+
 void c_render(Breakout* env) {
     if (env->client == NULL) {
         env->client = make_client(env);
@@ -582,6 +597,9 @@ void c_render(Breakout* env) {

     DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
     DrawText(TextFormat("Balls: %i", env->num_balls), client->width - 80, 10, 20, WHITE);
+    if (g_breakout_render_overlay != NULL) {
+        g_breakout_render_overlay(env, g_breakout_render_overlay_user);
+    }
     EndDrawing();

     //PlaySound(client->sound);
diff --git a/ocean/breakout/exporter.h b/ocean/breakout/exporter.h
new file mode 100644
index 00000000..d8c3de4c
--- /dev/null
+++ b/ocean/breakout/exporter.h
@@ -0,0 +1,376 @@
+#ifndef PUFFERLIB_BREAKOUT_EXPORTER_H
+#define PUFFERLIB_BREAKOUT_EXPORTER_H
+
+#include <errno.h>
+#include <float.h>
+#include <limits.h>
+#include <stdio.h>
+#include <stdlib.h>
+#include <string.h>
+
+#define PUFFERLIB_BREAKOUT_EXPORT_VERSION 1
+#define PUFFERLIB_BREAKOUT_OBS_DIM 118
+#define PUFFERLIB_BREAKOUT_ACTION_COUNT 3
+
+typedef struct BreakoutExportState {
+    int initialized;
+    int enabled;
+    int closed;
+    char jsonl_path[PATH_MAX];
+    char manifest_path[PATH_MAX];
+    char source[64];
+    long max_rows;
+    long row_count;
+    int env_capacity;
+    int* episodes;
+    int* timesteps;
+    float* returns;
+    int completed_episodes;
+    double completed_return_sum;
+    float min_return;
+    float max_return;
+    FILE* jsonl;
+} BreakoutExportState;
+
+static BreakoutExportState breakout_export_state;
+
+static void breakout_export_close(void);
+
+static void breakout_export_copy_source(char* dst, size_t cap, const char* src) {
+    if (src == NULL || src[0] == '\0') {
+        src = "train_export";
+    }
+
+    size_t n = 0;
+    while (src[n] != '\0' && n + 1 < cap) {
+        char c = src[n];
+        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
+                (c >= '0' && c <= '9') || c == '_' || c == '-') {
+            dst[n] = c;
+        } else {
+            dst[n] = '_';
+        }
+        n++;
+    }
+    dst[n] = '\0';
+}
+
+static void breakout_export_default_manifest_path(
+    char* dst, size_t cap, const char* jsonl_path
+) {
+    const char* slash = strrchr(jsonl_path, '/');
+    if (slash == NULL) {
+        snprintf(dst, cap, "manifest.json");
+        return;
+    }
+
+    size_t dir_len = (size_t)(slash - jsonl_path + 1);
+    if (dir_len >= cap) {
+        dir_len = cap - 1;
+    }
+    memcpy(dst, jsonl_path, dir_len);
+    dst[dir_len] = '\0';
+    strncat(dst, "manifest.json", cap - strlen(dst) - 1);
+}
+
+static void breakout_export_init(void) {
+    BreakoutExportState* st = &breakout_export_state;
+    if (st->initialized) {
+        return;
+    }
+
+    st->initialized = 1;
+    st->max_rows = -1;
+    st->min_return = FLT_MAX;
+    st->max_return = -FLT_MAX;
+
+    const char* jsonl = getenv("PUFFERLIB_BREAKOUT_EXPORT_JSONL");
+    if (jsonl == NULL || jsonl[0] == '\0') {
+        return;
+    }
+
+    snprintf(st->jsonl_path, sizeof(st->jsonl_path), "%s", jsonl);
+    const char* manifest = getenv("PUFFERLIB_BREAKOUT_EXPORT_MANIFEST");
+    if (manifest != NULL && manifest[0] != '\0') {
+        snprintf(st->manifest_path, sizeof(st->manifest_path), "%s", manifest);
+    } else {
+        breakout_export_default_manifest_path(
+            st->manifest_path, sizeof(st->manifest_path), st->jsonl_path);
+    }
+
+    breakout_export_copy_source(
+        st->source, sizeof(st->source),
+        getenv("PUFFERLIB_BREAKOUT_EXPORT_SOURCE"));
+
+    const char* max_rows = getenv("PUFFERLIB_BREAKOUT_EXPORT_MAX_ROWS");
+    if (max_rows != NULL && max_rows[0] != '\0') {
+        char* end = NULL;
+        errno = 0;
+        long parsed = strtol(max_rows, &end, 10);
+        if (errno == 0 && end != max_rows && parsed >= 0) {
+            st->max_rows = parsed;
+        }
+    }
+
+    st->jsonl = fopen(st->jsonl_path, "w");
+    if (st->jsonl == NULL) {
+        fprintf(stderr, "breakout exporter failed to open %s\n", st->jsonl_path);
+        return;
+    }
+
+    st->enabled = 1;
+    atexit(breakout_export_close);
+}
+
+static int breakout_export_enabled(void) {
+    breakout_export_init();
+    BreakoutExportState* st = &breakout_export_state;
+    return st->enabled && !st->closed &&
+        (st->max_rows < 0 || st->row_count < st->max_rows);
+}
+
+static int breakout_export_ensure_env(BreakoutExportState* st, int env_index) {
+    if (env_index < 0) {
+        return -1;
+    }
+    if (env_index < st->env_capacity) {
+        return 0;
+    }
+
+    int new_capacity = st->env_capacity == 0 ? 16 : st->env_capacity;
+    while (new_capacity <= env_index) {
+        new_capacity *= 2;
+    }
+
+    int* episodes = (int*)malloc((size_t)new_capacity * sizeof(int));
+    int* timesteps = (int*)malloc((size_t)new_capacity * sizeof(int));
+    float* returns = (float*)malloc((size_t)new_capacity * sizeof(float));
+    if (episodes == NULL || timesteps == NULL || returns == NULL) {
+        free(episodes);
+        free(timesteps);
+        free(returns);
+        return -1;
+    }
+
+    if (st->env_capacity > 0) {
+        memcpy(episodes, st->episodes, (size_t)st->env_capacity * sizeof(int));
+        memcpy(timesteps, st->timesteps, (size_t)st->env_capacity * sizeof(int));
+        memcpy(returns, st->returns, (size_t)st->env_capacity * sizeof(float));
+    }
+
+    for (int i = st->env_capacity; i < new_capacity; i++) {
+        episodes[i] = 0;
+        timesteps[i] = 0;
+        returns[i] = 0.0f;
+    }
+
+    free(st->episodes);
+    free(st->timesteps);
+    free(st->returns);
+    st->episodes = episodes;
+    st->timesteps = timesteps;
+    st->returns = returns;
+    st->env_capacity = new_capacity;
+    return 0;
+}
+
+static void breakout_export_record_episode(BreakoutExportState* st, float episode_return) {
+    if (episode_return < st->min_return) {
+        st->min_return = episode_return;
+    }
+    if (episode_return > st->max_return) {
+        st->max_return = episode_return;
+    }
+    st->completed_return_sum += episode_return;
+    st->completed_episodes++;
+}
+
+static void breakout_export_write_obs(FILE* f, const float* obs) {
+    fputs("\"obs\":[", f);
+    for (int i = 0; i < PUFFERLIB_BREAKOUT_OBS_DIM; i++) {
+        if (i > 0) {
+            fputc(',', f);
+        }
+        fprintf(f, "%.9g", obs[i]);
+    }
+    fputc(']', f);
+}
+
+static int breakout_export_transition(
+    int env_index,
+    const float obs[PUFFERLIB_BREAKOUT_OBS_DIM],
+    int action,
+    float reward,
+    int terminal
+) {
+    int rc = 0;
+    #pragma omp critical(pufferlib_breakout_export)
+    {
+        breakout_export_init();
+        BreakoutExportState* st = &breakout_export_state;
+        if (!st->enabled || st->closed ||
+                (st->max_rows >= 0 && st->row_count >= st->max_rows)) {
+            rc = 0;
+        } else if (obs == NULL || action < 0 ||
+                action >= PUFFERLIB_BREAKOUT_ACTION_COUNT ||
+                breakout_export_ensure_env(st, env_index) != 0) {
+            rc = -1;
+        } else {
+            int episode = st->episodes[env_index];
+            int t = st->timesteps[env_index];
+            int is_terminal = terminal ? 1 : 0;
+
+            fprintf(st->jsonl,
+                "{\"version\":%d,\"env\":\"pufferlib_breakout\","
+                "\"source\":\"%s\",\"env_index\":%d,\"episode\":%d,"
+                "\"t\":%d,\"obs_dim\":%d,",
+                PUFFERLIB_BREAKOUT_EXPORT_VERSION,
+                st->source,
+                env_index,
+                episode,
+                t,
+                PUFFERLIB_BREAKOUT_OBS_DIM);
+            breakout_export_write_obs(st->jsonl, obs);
+            fprintf(st->jsonl,
+                ",\"action\":%d,\"reward\":%.9g,\"terminal\":%d}\n",
+                action,
+                reward,
+                is_terminal);
+
+            st->row_count++;
+            st->returns[env_index] += reward;
+            st->timesteps[env_index]++;
+
+            if (is_terminal) {
+                breakout_export_record_episode(st, st->returns[env_index]);
+                st->returns[env_index] = 0.0f;
+                st->timesteps[env_index] = 0;
+                st->episodes[env_index]++;
+            }
+            rc = ferror(st->jsonl) ? -1 : 0;
+        }
+    }
+    return rc;
+}
+
+static void breakout_export_manifest_stats(
+    BreakoutExportState* st,
+    int* episode_count,
+    double* return_sum,
+    float* min_return,
+    float* max_return
+) {
+    *episode_count = st->completed_episodes;
+    *return_sum = st->completed_return_sum;
+    *min_return = st->min_return;
+    *max_return = st->max_return;
+
+    for (int i = 0; i < st->env_capacity; i++) {
+        if (st->timesteps[i] == 0) {
+            continue;
+        }
+        float episode_return = st->returns[i];
+        if (episode_return < *min_return) {
+            *min_return = episode_return;
+        }
+        if (episode_return > *max_return) {
+            *max_return = episode_return;
+        }
+        *return_sum += episode_return;
+        (*episode_count)++;
+    }
+
+    if (*episode_count == 0) {
+        *min_return = 0.0f;
+        *max_return = 0.0f;
+    }
+}
+
+static void breakout_export_close(void) {
+    #pragma omp critical(pufferlib_breakout_export)
+    {
+        BreakoutExportState* st = &breakout_export_state;
+        if (st->initialized && !st->closed) {
+            st->closed = 1;
+
+            if (st->jsonl != NULL) {
+                fclose(st->jsonl);
+                st->jsonl = NULL;
+            }
+
+            if (st->enabled) {
+            int episode_count = 0;
+            double return_sum = 0.0;
+            float min_return = 0.0f;
+            float max_return = 0.0f;
+            breakout_export_manifest_stats(
+                st, &episode_count, &return_sum, &min_return, &max_return);
+
+            FILE* manifest = fopen(st->manifest_path, "w");
+            if (manifest != NULL) {
+                double mean_return = episode_count > 0 ?
+                    return_sum / (double)episode_count : 0.0;
+                fprintf(manifest,
+                    "{\n"
+                    "  \"env\":\"pufferlib_breakout\",\n"
+                    "  \"source\":\"%s\",\n"
+                    "  \"version\":%d,\n"
+                    "  \"obs_dim\":%d,\n"
+                    "  \"action_count\":%d,\n"
+                    "  \"row_count\":%ld,\n"
+                    "  \"episode_count\":%d,\n"
+                    "  \"min_return\":%.9g,\n"
+                    "  \"max_return\":%.9g,\n"
+                    "  \"mean_return\":%.9g,\n"
+                    "  \"exporter\":{\n"
+                    "    \"jsonl_path\":\"%s\",\n"
+                    "    \"manifest_path\":\"%s\",\n"
+                    "    \"max_rows\":%ld\n"
+                    "  }\n"
+                    "}\n",
+                    st->source,
+                    PUFFERLIB_BREAKOUT_EXPORT_VERSION,
+                    PUFFERLIB_BREAKOUT_OBS_DIM,
+                    PUFFERLIB_BREAKOUT_ACTION_COUNT,
+                    st->row_count,
+                    episode_count,
+                    min_return,
+                    max_return,
+                    mean_return,
+                    st->jsonl_path,
+                    st->manifest_path,
+                    st->max_rows);
+                fclose(manifest);
+            }
+            }
+
+            free(st->episodes);
+            free(st->timesteps);
+            free(st->returns);
+            st->episodes = NULL;
+            st->timesteps = NULL;
+            st->returns = NULL;
+            st->env_capacity = 0;
+        }
+    }
+}
+
+static int breakout_export_step_env(Breakout* env, int env_index) {
+    if (!breakout_export_enabled()) {
+        c_step(env);
+        return 0;
+    }
+
+    float pre_obs[PUFFERLIB_BREAKOUT_OBS_DIM];
+    memcpy(pre_obs, env->observations, sizeof(pre_obs));
+    int action = (int)env->actions[0];
+    int action_observable = env->balls_fired != 0 || env->frameskip > 1;
+    c_step(env);
+    if (!action_observable) {
+        return 0;
+    }
+    return breakout_export_transition(
+        env_index, pre_obs, action, env->rewards[0], (int)env->terminals[0]);
+}
+
+#endif
diff --git a/ocean/breakout/tests/run_exporter_smoke.sh b/ocean/breakout/tests/run_exporter_smoke.sh
new file mode 100644
index 00000000..8294d820
--- /dev/null
+++ b/ocean/breakout/tests/run_exporter_smoke.sh
@@ -0,0 +1,24 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+cd "$(dirname "$0")/../../.."
+
+cc="${CC:-clang}"
+raylib_name="raylib-5.5_linux_amd64"
+raylib_a="$raylib_name/lib/libraylib.a"
+
+if [ ! -f "$raylib_a" ]; then
+    echo "missing $raylib_a; run ./build.sh breakout --local once to fetch raylib" >&2
+    exit 1
+fi
+
+mkdir -p build/breakout-tests
+"$cc" \
+    -std=gnu11 -Wall -Wextra -Werror=return-type -Wno-unused-function \
+    -I"$raylib_name/include" -Isrc -Iocean/breakout -Ivendor \
+    -DPLATFORM_DESKTOP \
+    ocean/breakout/tests/test_breakout_exporter.c \
+    "$raylib_a" -lGL -lm -lpthread -fopenmp \
+    -o build/breakout-tests/test_breakout_exporter
+
+build/breakout-tests/test_breakout_exporter
diff --git a/ocean/breakout/tests/test_breakout_exporter.c b/ocean/breakout/tests/test_breakout_exporter.c
new file mode 100644
index 00000000..8072fc1d
--- /dev/null
+++ b/ocean/breakout/tests/test_breakout_exporter.c
@@ -0,0 +1,125 @@
+#include <assert.h>
+#include <stdio.h>
+#include <stdlib.h>
+#include <string.h>
+
+#include "../breakout.h"
+#include "../exporter.h"
+
+static void require_contains(const char* text, const char* needle) {
+    if (strstr(text, needle) == NULL) {
+        fprintf(stderr, "missing expected text: %s\nin: %s\n", needle, text);
+        exit(1);
+    }
+}
+
+static void read_file(const char* path, char* buf, size_t cap) {
+    FILE* f = fopen(path, "rb");
+    assert(f != NULL);
+    size_t n = fread(buf, 1, cap - 1, f);
+    assert(!ferror(f));
+    fclose(f);
+    buf[n] = '\0';
+}
+
+int main(void) {
+    const char* dir = "/tmp/pufferlib_breakout_exporter_smoke";
+    const char* jsonl = "/tmp/pufferlib_breakout_exporter_smoke/transitions-000000.jsonl";
+    const char* manifest = "/tmp/pufferlib_breakout_exporter_smoke/manifest.json";
+
+    char cmd[512];
+    snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dir, dir);
+    assert(system(cmd) == 0);
+
+    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_JSONL", jsonl, 1) == 0);
+    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_MANIFEST", manifest, 1) == 0);
+    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_SOURCE", "train_export", 1) == 0);
+
+    Breakout env = {0};
+    env.num_agents = 1;
+    env.frameskip = 1;
+    env.width = 640;
+    env.height = 480;
+    env.initial_paddle_width = 80;
+    env.paddle_height = 20;
+    env.ball_width = 10;
+    env.ball_height = 10;
+    env.brick_width = 64;
+    env.brick_height = 20;
+    env.brick_rows = 6;
+    env.brick_cols = 18;
+    env.initial_ball_speed = 4;
+    env.max_ball_speed = 10;
+    env.paddle_speed = 8;
+    env.continuous = 0;
+    allocate(&env);
+
+    c_reset(&env);
+    env.actions[0] = 2.0f;
+    assert(breakout_export_step_env(&env, 3) == 0);
+    assert(env.balls_fired == 1);
+    assert(breakout_export_state.row_count == 0);
+
+    float pre_obs[118];
+    memcpy(pre_obs, env.observations, sizeof(pre_obs));
+    env.actions[0] = 1.0f;
+
+    assert(breakout_export_step_env(&env, 3) == 0);
+    assert(breakout_export_state.row_count == 1);
+
+    Breakout frameskip_env = {0};
+    frameskip_env.num_agents = 1;
+    frameskip_env.frameskip = 4;
+    frameskip_env.width = 640;
+    frameskip_env.height = 480;
+    frameskip_env.initial_paddle_width = 80;
+    frameskip_env.paddle_height = 20;
+    frameskip_env.ball_width = 10;
+    frameskip_env.ball_height = 10;
+    frameskip_env.brick_width = 64;
+    frameskip_env.brick_height = 20;
+    frameskip_env.brick_rows = 6;
+    frameskip_env.brick_cols = 18;
+    frameskip_env.initial_ball_speed = 4;
+    frameskip_env.max_ball_speed = 10;
+    frameskip_env.paddle_speed = 8;
+    frameskip_env.continuous = 0;
+    allocate(&frameskip_env);
+    c_reset(&frameskip_env);
+    frameskip_env.actions[0] = 2.0f;
+    assert(breakout_export_step_env(&frameskip_env, 4) == 0);
+    assert(breakout_export_state.row_count == 2);
+
+    breakout_export_close();
+
+    char row[16384];
+    read_file(jsonl, row, sizeof(row));
+    require_contains(row, "\"version\":1");
+    require_contains(row, "\"env\":\"pufferlib_breakout\"");
+    require_contains(row, "\"source\":\"train_export\"");
+    require_contains(row, "\"env_index\":3");
+    require_contains(row, "\"env_index\":4");
+    require_contains(row, "\"episode\":0");
+    require_contains(row, "\"t\":0");
+    require_contains(row, "\"obs_dim\":118");
+    require_contains(row, "\"action\":1");
+    require_contains(row, "\"action\":2");
+    require_contains(row, "\"reward\":");
+    require_contains(row, "\"terminal\":");
+
+    char first_obs[64];
+    snprintf(first_obs, sizeof(first_obs), "\"obs\":[%.9g", pre_obs[0]);
+    require_contains(row, first_obs);
+
+    char meta[4096];
+    read_file(manifest, meta, sizeof(meta));
+    require_contains(meta, "\"env\":\"pufferlib_breakout\"");
+    require_contains(meta, "\"source\":\"train_export\"");
+    require_contains(meta, "\"obs_dim\":118");
+    require_contains(meta, "\"action_count\":3");
+    require_contains(meta, "\"row_count\":2");
+
+    free_allocated(&frameskip_env);
+    free_allocated(&env);
+    return 0;
+}
diff --git a/ocean/craftax/binding.c b/ocean/craftax/binding.c
index bbed3259..16873959 100644
--- a/ocean/craftax/binding.c
+++ b/ocean/craftax/binding.c
@@ -12,6 +12,9 @@
 #define CRAFTAX_VEC_TILE_SIZE 128
 #define MY_VEC_INIT
 #define MY_VEC_CLOSE
+struct StaticVec;
+void craftax_vec_step(struct StaticVec* vec);
+void craftax_vec_step_range(struct StaticVec* vec, int env_start, int env_count, int num_workers);
 #define MY_VEC_STEP craftax_vec_step
 #define MY_VEC_STEP_RANGE craftax_vec_step_range
 #define Env Craftax
diff --git a/src/pufferlib.cu b/src/pufferlib.cu
index 583d9a11..6851e715 100644
--- a/src/pufferlib.cu
+++ b/src/pufferlib.cu
@@ -2201,9 +2201,15 @@ void close_impl(PuffeRL& pufferl) {
         cudaProfilerStop();
     }

-    cudaGraphExecDestroy(pufferl.train_cudagraph);
-    for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {
-        cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
+    if (pufferl.train_cudagraph != nullptr) {
+        cudaGraphExecDestroy(pufferl.train_cudagraph);
+    }
+    if (pufferl.fused_rollout_cudagraphs != nullptr) {
+        for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {
+            if (pufferl.fused_rollout_cudagraphs[i] != nullptr) {
+                cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
+            }
+        }
     }

     policy_weights_free(&pufferl.policy, &pufferl.weights);
diff --git a/src/vecenv.h b/src/vecenv.h
index 42958d32..8ae14d1b 100644
--- a/src/vecenv.h
+++ b/src/vecenv.h
@@ -258,8 +258,6 @@ static void* static_omp_threadmanager(void* arg) {
     int num_workers = threading->num_threads / vec->buffers;
     if (num_workers < 1) num_workers = 1;

-    Env* envs = (Env*)vec->envs;
-
     printf("Num workers: %d\n", num_workers);
     while (true) {
         while (atomic_load(&buffer_states[buf]) != OMP_RUNNING) {
@@ -288,10 +286,15 @@ static void* static_omp_threadmanager(void* arg) {
             memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
             memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
             clock_gettime(CLOCK_MONOTONIC, &t0);
-            #pragma omp parallel for schedule(static) num_threads(num_workers)
-            for (int i = env_start; i < env_start + env_count; i++) {
-                c_step(&envs[i]);
-            }
+            #ifdef MY_VEC_STEP_RANGE
+                MY_VEC_STEP_RANGE(vec, env_start, env_count, num_workers);
+            #else
+                Env* envs = (Env*)vec->envs;
+                #pragma omp parallel for schedule(static) num_threads(num_workers)
+                for (int i = env_start; i < env_start + env_count; i++) {
+                    c_step(&envs[i]);
+                }
+            #endif
             clock_gettime(CLOCK_MONOTONIC, &t1);
             my_accum[EVAL_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

@@ -739,6 +742,9 @@ const char* get_obs_dtype(void) { return dtype_symbol; }
 size_t get_obs_elem_size(void) { return obs_element_size(); }

 static inline void _static_vec_env_step(StaticVec* vec) {
+    #ifdef MY_VEC_STEP
+        MY_VEC_STEP(vec);
+    #else
     memset(vec->rewards, 0, vec->total_agents * sizeof(float));
     memset(vec->terminals, 0, vec->total_agents * sizeof(float));
     Env* envs = (Env*)vec->envs;
@@ -746,6 +752,7 @@ static inline void _static_vec_env_step(StaticVec* vec) {
     for (int i = 0; i < vec->size; i++) {
         c_step(&envs[i]);
     }
+    #endif
 }

 void gpu_vec_step(StaticVec* vec) {
```
