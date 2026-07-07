#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
typedef int cudaMemcpyKind;

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    (void)flags;
    *ptr = calloc(1, size);
    return *ptr ? 0 : 1;
}

cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = calloc(1, size);
    return *ptr ? 0 : 1;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, size);
    return 0;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy(dst, src, size, kind);
}

cudaError_t cudaMemset(void* ptr, int value, size_t size) {
    memset(ptr, value, size);
    return 0;
}

cudaError_t cudaFree(void* ptr) {
    free(ptr);
    return 0;
}

cudaError_t cudaFreeHost(void* ptr) {
    free(ptr);
    return 0;
}

cudaError_t cudaSetDevice(int device) {
    (void)device;
    return 0;
}

cudaError_t cudaDeviceSynchronize(void) {
    return 0;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return 0;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void)flags;
    *stream = NULL;
    return 0;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    (void)stream;
    return 0;
}

const char* cudaGetErrorString(cudaError_t error) {
    (void)error;
    return "test cuda stub";
}

#include "/home/claude/pathfinder/ocean/breakout/binding.c"

static void require_contains(const char* text, const char* needle) {
    if (strstr(text, needle) == NULL) {
        fprintf(stderr, "missing expected text: %s\nin: %s\n", needle, text);
        exit(1);
    }
}

static void read_file(const char* path, char* buf, size_t cap) {
    FILE* f = fopen(path, "rb");
    size_t n;

    assert(f != NULL);
    n = fread(buf, 1, cap - 1u, f);
    assert(!ferror(f));
    fclose(f);
    buf[n] = '\0';
}

static void init_env(Breakout* env, int frameskip, unsigned int seed) {
    memset(env, 0, sizeof(*env));
    env->num_agents = 1;
    env->frameskip = frameskip;
    env->width = 576;
    env->height = 330;
    env->initial_paddle_width = 62;
    env->paddle_width = 62;
    env->paddle_height = 8;
    env->ball_width = 32;
    env->ball_height = 32;
    env->brick_width = 32;
    env->brick_height = 12;
    env->brick_rows = 6;
    env->brick_cols = 18;
    env->initial_ball_speed = 256;
    env->max_ball_speed = 448;
    env->paddle_speed = 620;
    env->continuous = 0;
    env->rng = seed;
    allocate(env);
    c_reset(env);
}

int main(void) {
    const char* jsonl = "/tmp/tkm_train_export_hook_smoke/transitions-000000.jsonl";
    const char* manifest = "/tmp/tkm_train_export_hook_smoke/manifest.json";
    Breakout envs[2];
    StaticVec vec;
    char text[32768];

    assert(system("rm -rf /tmp/tkm_train_export_hook_smoke && mkdir -p /tmp/tkm_train_export_hook_smoke") == 0);
    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_JSONL", jsonl, 1) == 0);
    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_MANIFEST", manifest, 1) == 0);
    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_SOURCE", "train_export", 1) == 0);
    assert(setenv("PUFFERLIB_BREAKOUT_EXPORT_MAX_ROWS", "4", 1) == 0);

    init_env(&envs[0], 4, 1u);
    init_env(&envs[1], 4, 2u);
    envs[0].actions[0] = 1.0f;
    envs[1].actions[0] = 2.0f;

    memset(&vec, 0, sizeof(vec));
    vec.envs = envs;
    vec.size = 2;
    vec.total_agents = 2;

    breakout_vec_step_range(&vec, 0, 2, 1);
    breakout_vec_step_range(&vec, 0, 2, 1);
    breakout_export_close();

    read_file(jsonl, text, sizeof(text));
    require_contains(text, "\"source\":\"train_export\"");
    require_contains(text, "\"env_index\":0");
    require_contains(text, "\"env_index\":1");
    require_contains(text, "\"action\":1");
    require_contains(text, "\"action\":2");
    require_contains(text, "\"obs_dim\":118");
    require_contains(text, "\"reward\":");
    require_contains(text, "\"terminal\":");

    read_file(manifest, text, sizeof(text));
    require_contains(text, "\"source\":\"train_export\"");
    require_contains(text, "\"row_count\":4");
    require_contains(text, "\"obs_dim\":118");

    free_allocated(&envs[1]);
    free_allocated(&envs[0]);
    return 0;
}
