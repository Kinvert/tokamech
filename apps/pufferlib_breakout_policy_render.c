#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "breakout.h"

#include "core/dataset/float_transition.h"
#include "projects/breakout/pufferlib_policy.h"

#ifndef PUFFERLIB_DIR
#define PUFFERLIB_DIR "/home/claude/pathfinder"
#endif

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

static int32_t read_i32_env(const char* name, int32_t fallback) {
    const char* text = getenv(name);
    char* end = NULL;
    long value;

    if (!text || text[0] == '\0') {
        return fallback;
    }
    value = strtol(text, &end, 10);
    if (end == text || value < -1000000000L || value > 1000000000L) {
        return fallback;
    }
    return (int32_t)value;
}

static Breakout make_pufferlib_breakout_env(void) {
    Breakout env = {
        .client = NULL,
        .num_agents = 1,
        .frameskip = (int)read_u32_env("TKM_PUFFERLIB_BREAKOUT_FRAMESKIP", 4u),
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

int main(void) {
    const char* config_path = getenv("TKM_PUFFERLIB_POLICY_CONFIG");
    const char* params_path = getenv("TKM_PUFFERLIB_POLICY_PARAMS");
    const char* dataset_path = getenv("TKM_PUFFERLIB_BREAKOUT_JSONL");
    const char* mode = getenv("TKM_PUFFERLIB_RENDER_MODE");
    const char* max_steps_text = getenv("TKM_PUFFERLIB_RENDER_MAX_STEPS");
    BreakoutPufferlibPolicy policy;
    BreakoutPufferlibTokenPolicy token_policy;
    BreakoutPufferlibTokenConfig token_config;
    BreakoutPufferlibTokenReport token_report;
    static TkmFloatTransitionDataset dataset;
    Breakout env;
    uint32_t action_histogram[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT] = {0u, 0u, 0u};
    uint32_t full_searches = 0u;
    uint32_t sequence_hits = 0u;
    uint32_t window_searches = 0u;
    uint32_t steps = 0;
    uint32_t max_steps = 0;
    uint32_t cursor = 0;
    uint32_t window_radius = read_u32_env("TKM_PUFFERLIB_BREAKOUT_WINDOW_RADIUS", 2048u);
    uint32_t reanchor_interval = read_u32_env("TKM_PUFFERLIB_BREAKOUT_REANCHOR_INTERVAL", 32u);
    float sequence_distance = (float)read_u32_env("TKM_PUFFERLIB_BREAKOUT_SEQUENCE_DISTANCE_MILLI", 500u) / 1000.0f;
    float intercept_deadzone = (float)read_u32_env("TKM_PUFFERLIB_BREAKOUT_INTERCEPT_DEADZONE_MILLI", 30u) / 1000.0f;
    float intercept_aim = (float)read_i32_env("TKM_PUFFERLIB_BREAKOUT_INTERCEPT_AIM_MILLI", 0) / 1000.0f;
    uint32_t token_bins = read_u32_env("TKM_PUFFERLIB_BREAKOUT_TOKEN_BINS", 16u);
    uint32_t token_dims = read_u32_env("TKM_PUFFERLIB_BREAKOUT_TOKEN_DIMS", 16u);
    uint32_t token_epochs = read_u32_env("TKM_PUFFERLIB_BREAKOUT_TOKEN_EPOCHS", 80u);
    BreakoutPufferlibSequenceCursor sequence_cursor;
    BreakoutPufferlibInterceptPolicy intercept_policy;
    BreakoutPufferlibInterceptReport intercept_report;
    uint8_t has_cursor = 0;
    uint8_t action = NOOP;
    int use_mlp = 0;
    int use_token = 1;
    int use_sequence = 0;
    int use_nearest = 0;
    int use_intercept = 0;
    int use_intercept_trained = 0;

    if (!config_path || config_path[0] == '\0') {
        config_path = "/tmp/tkm_pufferlib_breakout_smoke/tokamech_policy.ini";
    }
    if (!params_path || params_path[0] == '\0') {
        params_path = "/tmp/tkm_pufferlib_breakout_smoke/tokamech_policy.params";
    }
    if (!dataset_path || dataset_path[0] == '\0') {
        dataset_path = "/tmp/tkm_pufferlib_breakout_smoke/transitions-000000.jsonl";
    }
    if (mode && strcmp(mode, "mlp") == 0) {
        use_mlp = 1;
        use_token = 0;
        use_sequence = 0;
        use_nearest = 0;
        use_intercept = 0;
        use_intercept_trained = 0;
    } else if (mode && (strcmp(mode, "token") == 0 || strcmp(mode, "token_mlp") == 0)) {
        use_mlp = 0;
        use_token = 1;
        use_sequence = 0;
        use_nearest = 0;
        use_intercept = 0;
        use_intercept_trained = 0;
    } else if (mode && strcmp(mode, "nearest") == 0) {
        use_mlp = 0;
        use_token = 0;
        use_sequence = 0;
        use_nearest = 1;
        use_intercept = 0;
        use_intercept_trained = 0;
    } else if (mode && (strcmp(mode, "sequence") == 0 || strcmp(mode, "sequence_cursor") == 0)) {
        use_mlp = 0;
        use_token = 0;
        use_sequence = 1;
        use_nearest = 0;
        use_intercept = 0;
        use_intercept_trained = 0;
    } else if (mode && strcmp(mode, "intercept_rule") == 0) {
        use_mlp = 0;
        use_token = 0;
        use_sequence = 0;
        use_nearest = 0;
        use_intercept = 1;
        use_intercept_trained = 0;
    } else if (mode && strcmp(mode, "intercept_trained") == 0) {
        use_mlp = 0;
        use_token = 0;
        use_sequence = 0;
        use_nearest = 0;
        use_intercept = 0;
        use_intercept_trained = 1;
    }
    if (max_steps_text && max_steps_text[0] != '\0') {
        max_steps = (uint32_t)strtoul(max_steps_text, NULL, 10);
    }

    if (!use_mlp && (use_token || use_sequence || use_nearest || use_intercept_trained)) {
        fail_if(tkm_float_transition_dataset_load_jsonl(&dataset, dataset_path),
            "tkm_float_transition_dataset_load_jsonl");
        if (use_token) {
            token_config.bin_count = token_bins;
            token_config.selected_dim_count = token_dims;
            token_config.epochs = token_epochs;
            token_config.learning_rate = 0.20f;
            token_config.heldout_stride = 5u;
            token_config.use_pair_features = 1u;
            fail_if(breakout_pufferlib_token_policy_train(
                    &token_policy, &dataset, &token_config, &token_report),
                "breakout_pufferlib_token_policy_train");
            printf("pufferlib breakout token_mlp rows=%u heldout_accuracy=%.3f bins=%u dims=%u epochs=%u input_dim=%u hidden=%u\n",
                token_report.source_rows,
                token_report.heldout_accuracy,
                token_report.bin_count,
                token_report.selected_dim_count,
                token_epochs,
                token_policy.token_mlp_input_dim,
                token_policy.token_mlp_hidden_dim);
        } else if (use_sequence) {
            fail_if(breakout_pufferlib_sequence_cursor_init(
                    &sequence_cursor, reanchor_interval, sequence_distance),
                "breakout_pufferlib_sequence_cursor_init");
        } else if (use_intercept_trained) {
            fail_if(breakout_pufferlib_intercept_policy_train_grid(
                    &intercept_policy, &dataset, &intercept_report),
                "breakout_pufferlib_intercept_policy_train_grid");
            printf("pufferlib breakout trained_intercept rows=%u heldout_accuracy=%.3f deadzone=%.3f aim_offset=%.3f\n",
                intercept_report.source_rows,
                intercept_report.heldout_accuracy,
                intercept_report.deadzone,
                intercept_report.aim_offset);
        }
    } else {
        if (use_mlp) {
            fail_if(breakout_pufferlib_policy_load(&policy, config_path, params_path),
                "breakout_pufferlib_policy_load");
        } else if (use_intercept) {
            fail_if(breakout_pufferlib_intercept_policy_init(
                    &intercept_policy, intercept_deadzone, intercept_aim),
                "breakout_pufferlib_intercept_policy_init");
        }
    }

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
            has_cursor = 0;
            if (use_token) {
                breakout_pufferlib_token_policy_reset(&token_policy);
            }
            if (use_sequence) {
                breakout_pufferlib_sequence_cursor_reset(&sequence_cursor);
            }
        }

        if (env.balls_fired == 0 && env.frameskip == 1) {
            action = NOOP;
        } else if (!use_mlp) {
            uint32_t found_index = 0;
            uint8_t use_full;
            uint32_t start = 0;
            uint32_t count = dataset.row_count;

            if (use_token) {
                fail_if(breakout_pufferlib_token_policy_predict(
                        &token_policy, env.observations, &action),
                    "breakout_pufferlib_token_policy_predict");
                goto have_action;
            }

            if (use_intercept) {
                fail_if(breakout_pufferlib_intercept_policy_predict(
                        &intercept_policy, env.observations, &action),
                    "breakout_pufferlib_intercept_policy_predict");
                goto have_action;
            }

            if (use_sequence) {
                BreakoutPufferlibSequenceMatch match;
                fail_if(breakout_pufferlib_sequence_cursor_predict(
                        &dataset,
                        &sequence_cursor,
                        env.observations,
                        steps,
                        &action,
                        &match,
                        NULL,
                        NULL),
                    "breakout_pufferlib_sequence_cursor_predict");
                if (match == BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_FULL_SEARCH) {
                    full_searches++;
                } else {
                    sequence_hits++;
                }
                goto have_action;
            }

            use_full = !has_cursor ||
                cursor + 1u >= dataset.row_count ||
                (reanchor_interval > 0u && steps % reanchor_interval == 0u);
            if (use_nearest && !use_full) {
                uint32_t next_cursor = cursor + 1u;
                start = next_cursor > window_radius ? next_cursor - window_radius : 0u;
                count = window_radius * 2u + 1u;
                if (start + count > dataset.row_count) {
                    count = dataset.row_count - start;
                }
            }

            fail_if(breakout_pufferlib_nearest_predict_range(
                    &dataset, env.observations, start, count, &action, &found_index, NULL),
                "breakout_pufferlib_nearest_predict_range");
            cursor = found_index;
            has_cursor = 1u;
            if (use_full) {
                full_searches++;
            } else {
                window_searches++;
            }
        } else {
            fail_if(breakout_pufferlib_policy_predict(&policy, env.observations, &action),
                "breakout_pufferlib_policy_predict");
        }

have_action:
        env.actions[0] = (float)action;
        if (action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
            action_histogram[action]++;
        }
        c_step(&env);
        c_render(&env);
        steps++;
        if (env.terminals[0] != 0.0f) {
            has_cursor = 0u;
            if (use_token) {
                breakout_pufferlib_token_policy_reset(&token_policy);
            }
            if (use_sequence) {
                breakout_pufferlib_sequence_cursor_reset(&sequence_cursor);
            }
        }

        if (max_steps > 0u && steps >= max_steps) {
            break;
        }
    }

    close_client(env.client);
    free_allocated(&env);
    {
        const char* mode_name = "intercept_rule";
        if (use_mlp) {
            mode_name = "mlp";
        } else if (use_token) {
            mode_name = "token_mlp";
        } else if (use_sequence) {
            mode_name = "sequence_cursor";
        } else if (use_nearest) {
            mode_name = "nearest_window";
        } else if (use_intercept_trained) {
            mode_name = "intercept_trained";
        }

    printf("pufferlib breakout render mode=%s steps=%u action_histogram=[%u,%u,%u] full_searches=%u sequence_hits=%u window_searches=%u\n",
        mode_name,
        steps,
        action_histogram[0],
        action_histogram[1],
        action_histogram[2],
        full_searches,
        sequence_hits,
        window_searches
    );
    }
    return 0;
}
