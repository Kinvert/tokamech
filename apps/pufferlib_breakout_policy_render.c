#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "breakout.h"

#include "core/dataset/float_transition.h"
#include "projects/breakout/pufferlib_policy.h"

#ifndef PUFFERLIB_DIR
#define PUFFERLIB_DIR "/home/claude/pathfinder"
#endif

#define DEFAULT_TOKEN_CONFIG_PATH "projects/breakout/pufferlib_token_ngram.ini"

static uint32_t g_configured_frameskip = 4u;

static void fail_if(int status, const char* label) {
    if (status != TKM_OK) {
        fprintf(stderr, "%s failed\n", label);
        exit(1);
    }
}

static uint32_t read_u32_env(const char* name, uint32_t fallback) {
    const char* text = getenv(name);
    char* end = NULL;
    unsigned long value;

    if (!text || text[0] == '\0') {
        return fallback;
    }
    value = strtoul(text, &end, 10);
    if (end == text || value > 0xfffffffful) {
        return fallback;
    }
    return (uint32_t)value;
}

static int read_ini_u32(const TkmIni* ini, const char* section, const char* key, uint32_t fallback, uint32_t* out) {
    int32_t parsed = 0;

    if (!out) {
        return TKM_ERR;
    }
    if (tkm_ini_get_i32(ini, section, key, (int32_t)fallback, &parsed) != TKM_OK || parsed < 0) {
        return TKM_ERR;
    }
    *out = (uint32_t)parsed;
    return TKM_OK;
}

static int load_token_config(
    const char* path,
    TkmIni* ini,
    BreakoutPufferlibTokenConfig* config,
    const char** out_dataset_path,
    uint32_t* out_frameskip
) {
    uint32_t frameskip = 4u;

    if (!path || !ini || !config || !out_dataset_path || !out_frameskip) {
        return TKM_ERR;
    }
    if (tkm_ini_parse_file(ini, path) != TKM_OK ||
        breakout_pufferlib_token_config_from_ini(ini, config) != TKM_OK ||
        read_ini_u32(ini, "breakout.pufferlib", "frameskip", 4u, &frameskip) != TKM_OK ||
        frameskip == 0u) {
        return TKM_ERR;
    }
    *out_dataset_path = tkm_ini_get(ini, "breakout.pufferlib", "dataset", "");
    *out_frameskip = frameskip;
    return TKM_OK;
}

static Breakout make_pufferlib_breakout_env(void) {
    Breakout env = {
        .client = NULL,
        .num_agents = 1,
        .frameskip = (int)read_u32_env("TKM_PUFFERLIB_BREAKOUT_FRAMESKIP", g_configured_frameskip),
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
        .rng = 1u,
    };
    allocate(&env);
    c_reset(&env);
    return env;
}

static const char* token_mode_name(const BreakoutPufferlibTokenReport* report) {
    if (strcmp(report->sequence_model_name, "token_backoff_ngram") == 0) {
        return "token_backoff_ngram";
    }
    if (strcmp(report->sequence_model_name, "token_mlp_window") == 0) {
        return "token_mlp_window";
    }
    if (strcmp(report->sequence_model_name, "token_ngram") == 0) {
        return "token_ngram";
    }
    return "token_policy";
}

int main(void) {
    const char* token_config_path = getenv("TKM_PUFFERLIB_BREAKOUT_CONFIG");
    const char* dataset_path = getenv("TKM_PUFFERLIB_BREAKOUT_JSONL");
    const char* config_dataset_path = "";
    const char* max_steps_text = getenv("TKM_PUFFERLIB_RENDER_MAX_STEPS");
    BreakoutPufferlibTokenPolicy token_policy;
    BreakoutPufferlibTokenConfig token_config;
    BreakoutPufferlibTokenReport token_report;
    TkmIni token_ini;
    static TkmFloatTransitionDataset dataset;
    Breakout env;
    uint32_t action_histogram[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT] = {0u, 0u, 0u};
    uint32_t steps = 0;
    uint32_t max_steps = 0;
    uint8_t action = NOOP;

    if (max_steps_text && max_steps_text[0] != '\0') {
        max_steps = (uint32_t)strtoul(max_steps_text, NULL, 10);
    }
    if (!token_config_path || token_config_path[0] == '\0') {
        token_config_path = DEFAULT_TOKEN_CONFIG_PATH;
    }
    fail_if(load_token_config(
            token_config_path,
            &token_ini,
            &token_config,
            &config_dataset_path,
            &g_configured_frameskip),
        "load_token_config");
    if (!dataset_path || dataset_path[0] == '\0') {
        dataset_path = config_dataset_path;
    }
    if (!dataset_path || dataset_path[0] == '\0') {
        dataset_path = "/tmp/tkm_pufferlib_breakout_smoke/transitions-000000.jsonl";
    }

    fail_if(tkm_float_transition_dataset_load_jsonl(&dataset, dataset_path),
        "tkm_float_transition_dataset_load_jsonl");
    fail_if(breakout_pufferlib_token_policy_train(
            &token_policy, &dataset, &token_config, &token_report),
        "breakout_pufferlib_token_policy_train");
    printf("pufferlib breakout token_stack config=%s dataset=%s frameskip=%u\n",
        token_config_path,
        dataset_path,
        g_configured_frameskip);
    printf("pufferlib breakout token_layers layout=%s tokenizer=%s selection=%s input=%s model=%s head=%s\n",
        token_report.sequence_layout_name,
        token_report.observation_tokenizer_name,
        token_report.observation_selection_name,
        token_report.input_feature_name,
        token_report.sequence_model_name,
        token_report.action_head_name);
    printf("pufferlib breakout token_train rows=%u heldout_accuracy=%.3f obs_token_accuracy=%.3f action_token_accuracy=%.3f exact=%u backoff=%u prior=%u bins=%u dims=%u history=%u epochs=%u learning_rate=%.5f input_dim=%u hidden=%u\n",
        token_report.source_rows,
        token_report.heldout_accuracy,
        token_report.obs_token_accuracy,
        token_report.action_token_accuracy,
        token_report.exact_predictions,
        token_report.backoff_predictions,
        token_report.prior_predictions,
        token_report.bin_count,
        token_report.selected_dim_count,
        token_policy.history,
        token_config.epochs,
        token_config.learning_rate,
        token_policy.token_mlp_input_dim,
        token_policy.token_mlp_hidden_dim);

    if (chdir(PUFFERLIB_DIR) != 0) {
        perror("chdir PUFFERLIB_DIR");
        return 1;
    }

    env = make_pufferlib_breakout_env();
    env.client = make_client(&env);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            action = NOOP;
            breakout_pufferlib_token_policy_reset(&token_policy);
        }

        if (env.balls_fired == 0 && env.frameskip == 1) {
            action = NOOP;
        } else {
            fail_if(breakout_pufferlib_token_policy_predict(
                    &token_policy, env.observations, &action),
                "breakout_pufferlib_token_policy_predict");
        }

        env.actions[0] = (float)action;
        if (action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
            action_histogram[action]++;
        }
        c_step(&env);
        c_render(&env);
        steps++;
        if (env.terminals[0] != 0.0f) {
            breakout_pufferlib_token_policy_reset(&token_policy);
        }

        if (max_steps > 0u && steps >= max_steps) {
            break;
        }
    }

    close_client(env.client);
    free_allocated(&env);
    printf("pufferlib breakout render mode=%s steps=%u action_histogram=[%u,%u,%u]\n",
        token_mode_name(&token_report),
        steps,
        action_histogram[0],
        action_histogram[1],
        action_histogram[2]
    );
    return 0;
}
