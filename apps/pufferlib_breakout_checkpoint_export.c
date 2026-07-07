#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "breakout.h"
#include "exporter.h"
#include "puffernet.h"

static int read_int_env(const char* name, int fallback) {
    const char* text = getenv(name);
    char* end = NULL;
    long value;

    if (!text || text[0] == '\0') {
        return fallback;
    }
    value = strtol(text, &end, 10);
    if (end == text || value <= 0 || value > 1000000000L) {
        return fallback;
    }
    return (int)value;
}

static void forward_puffernet_greedy(PufferNet* net, float* observations, float* actions) {
    linear(net->encoder, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    if (net->is_continuous) {
        _gaussian_mean(net->decoder->output, actions, net->num_agents, net->num_actions);
    } else {
        argmax_multidiscrete(net->multidiscrete, net->decoder->output, actions);
    }
}

static Breakout make_export_env(void) {
    Breakout env = {
        .client = NULL,
        .num_agents = 1,
        .frameskip = read_int_env("TKM_PUFFERLIB_BREAKOUT_FRAMESKIP", 4),
        .width = 576,
        .height = 330,
        .initial_paddle_width = 62,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .initial_ball_speed = 256,
        .max_ball_speed = 448,
        .paddle_speed = 620,
        .continuous = 0,
        .rng = (unsigned int)read_int_env("TKM_PUFFERLIB_BREAKOUT_SEED", 0),
    };
    allocate(&env);
    c_reset(&env);
    return env;
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : NULL;
    int steps = argc > 2 ? atoi(argv[2]) : read_int_env("TKM_PUFFERLIB_BREAKOUT_EXPORT_STEPS", 200000);
    int greedy = read_int_env("TKM_PUFFERLIB_BREAKOUT_EXPORT_GREEDY", 0);
    int logit_sizes[1] = {3};
    Weights* weights;
    PufferNet* net;
    Breakout env;

    if (!weights_path || weights_path[0] == '\0' || steps <= 0) {
        fprintf(stderr, "usage: %s WEIGHTS_PATH [STEPS]\n", argv[0]);
        return 1;
    }
    setenv("PUFFERLIB_BREAKOUT_EXPORT_SOURCE",
        greedy ? "trained_checkpoint_greedy_export" : "trained_checkpoint_export", 0);

    weights = load_weights(weights_path);
    if (!weights) {
        return 1;
    }
    net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);
    env = make_export_env();

    for (int step = 0; step < steps; step++) {
        if (greedy) {
            forward_puffernet_greedy(net, env.observations, env.actions);
        } else {
            forward_puffernet(net, env.observations, env.actions);
        }
        if (breakout_export_step_env(&env, 0) != 0) {
            free_puffernet(net);
            free(weights);
            free_allocated(&env);
            return 1;
        }
    }

    breakout_export_close();
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    return 0;
}
