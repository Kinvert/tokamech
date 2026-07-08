#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "breakout.h"

#include "core/dataset/float_transition.h"
#include "projects/breakout/pufferlib_policy.h"

#define DEFAULT_EVAL_EPISODES 2u
#define DEFAULT_MAX_STEPS 24000u
#define DEFAULT_WINDOW_RADIUS 2048u
#define DEFAULT_REANCHOR_INTERVAL 32u
#define DEFAULT_TOKEN_CONFIG_PATH "projects/breakout/pufferlib_token_ngram.ini"

static uint32_t g_configured_frameskip = 1u;

typedef enum {
    BENCH_METHOD_TOKEN = 0
} BenchMethod;

typedef struct {
    BenchMethod method;
    const TkmFloatTransitionDataset* dataset;
    BreakoutPufferlibTokenPolicy* token_policy;
} BenchPolicy;

typedef struct {
    char name[64];
    uint32_t rows;
    uint32_t episodes;
    uint32_t steps;
    uint32_t full_searches;
    uint32_t window_searches;
    uint32_t action_histogram[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float return_sum;
    float length_sum;
    float max_return;
} BenchResult;

static float obs_distance(const float* a, const float* b);

static uint32_t read_u32_env(const char* name, uint32_t fallback) {
    const char* value = getenv(name);
    char* end = NULL;
    unsigned long parsed;

    if (!value || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || parsed > 0xfffffffful) {
        return fallback;
    }
    return (uint32_t)parsed;
}

static int make_dir_if_needed(const char* path) {
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return TKM_OK;
    }
    return TKM_ERR;
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

static const char* token_result_name(const BreakoutPufferlibTokenReport* report, uint8_t top1) {
    if (report && strcmp(report->sequence_model_name, "token_backoff_ngram") == 0) {
        return top1 ? "token_backoff_ngram_top1" : "token_backoff_ngram_all";
    }
    if (report && strcmp(report->sequence_model_name, "token_mlp_window") == 0) {
        return top1 ? "token_mlp_window_top1" : "token_mlp_window_all";
    }
    if (report && strcmp(report->sequence_model_name, "token_ngram") == 0) {
        return top1 ? "token_ngram_top1" : "token_ngram_all";
    }
    if (report && strcmp(report->observation_selection_name, "first_dims") == 0) {
        return top1 ? "token_mlp_first_dims_top1" : "token_mlp_first_dims_all";
    }
    if (report && strcmp(report->observation_selection_name, "manual") == 0) {
        return top1 ? "token_mlp_manual_top1" : "token_mlp_manual_all";
    }
    return top1 ? "token_mlp_top1" : "token_mlp_all";
}

static Breakout make_eval_env(uint32_t seed) {
    Breakout env = {
        .client = NULL,
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
        .rng = seed,
    };
    allocate(&env);
    c_reset(&env);
    return env;
}

static int policy_action(
    BenchPolicy* policy,
    const float* obs,
    uint32_t step,
    uint8_t* out_action,
    BenchResult* result
) {
    (void)step;
    (void)result;
    if (!policy || policy->method != BENCH_METHOD_TOKEN || !obs || !out_action) {
        return TKM_ERR;
    }
    return breakout_pufferlib_token_policy_predict(policy->token_policy, obs, out_action);
}

static int eval_policy(
    const char* name,
    BenchPolicy* policy,
    uint32_t seed,
    uint32_t target_episodes,
    uint32_t max_steps,
    BenchResult* result
) {
    Breakout env;
    float episode_return = 0.0f;
    uint32_t episode_length = 0u;

    if (!name || !policy || !result || target_episodes == 0u || max_steps == 0u) {
        return TKM_ERR;
    }

    memset(result, 0, sizeof(*result));
    snprintf(result->name, sizeof(result->name), "%s", name);
    result->rows = policy->dataset ? policy->dataset->row_count : 0u;
    env = make_eval_env(seed);
    if (breakout_pufferlib_token_policy_reset(policy->token_policy) != TKM_OK) {
        free_allocated(&env);
        return TKM_ERR;
    }

    while (result->episodes < target_episodes && result->steps < max_steps) {
        uint8_t action = NOOP;

        if (env.balls_fired != 0 || env.frameskip > 1) {
            if (policy_action(policy, env.observations, result->steps, &action, result) != TKM_OK) {
                free_allocated(&env);
                return TKM_ERR;
            }
        }

        if (action >= TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
            free_allocated(&env);
            return TKM_ERR;
        }
        env.actions[0] = (float)action;
        result->action_histogram[action]++;
        c_step(&env);

        episode_return += env.rewards[0];
        episode_length++;
        result->steps++;

        if (env.terminals[0] != 0.0f) {
            result->episodes++;
            result->return_sum += episode_return;
            result->length_sum += (float)episode_length;
            if (episode_return > result->max_return || result->episodes == 1u) {
                result->max_return = episode_return;
            }
            episode_return = 0.0f;
            episode_length = 0u;
            breakout_pufferlib_token_policy_reset(policy->token_policy);
        }
    }

    if (episode_length > 0u && result->episodes == 0u) {
        result->episodes = 1u;
        result->return_sum = episode_return;
        result->length_sum = (float)episode_length;
        result->max_return = episode_return;
    }

    free_allocated(&env);
    return TKM_OK;
}

static void print_result(const BenchResult* result) {
    float mean_return = result->episodes > 0u ? result->return_sum / (float)result->episodes : 0.0f;
    float mean_length = result->episodes > 0u ? result->length_sum / (float)result->episodes : 0.0f;

    printf("method=%s rows=%u episodes=%u mean_return=%.3f max_return=%.3f mean_length=%.3f steps=%u actions=[%u,%u,%u] full_searches=%u window_searches=%u\n",
        result->name,
        result->rows,
        result->episodes,
        mean_return,
        result->max_return,
        mean_length,
        result->steps,
        result->action_histogram[0],
        result->action_histogram[1],
        result->action_histogram[2],
        result->full_searches,
        result->window_searches);
}

static float obs_distance(const float* a, const float* b) {
    float distance = 0.0f;
    for (uint32_t i = 0; i < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; i++) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

static int eval_dataset_replay(
    const TkmFloatTransitionDataset* dataset,
    uint32_t seed,
    uint32_t max_steps,
    BenchResult* result,
    float* out_max_obs_distance
) {
    Breakout env;
    float episode_return = 0.0f;
    uint32_t episode_length = 0u;
    uint32_t row_index = 0u;
    float max_obs_distance = 0.0f;
    float first_obs_distance = -1.0f;
    uint32_t first_drift_row = 0xffffffffu;
    float first_drift_distance = 0.0f;
    uint32_t first_reward_mismatch_row = 0xffffffffu;

    if (!dataset || !result || !out_max_obs_distance || max_steps == 0u) {
        return TKM_ERR;
    }
    memset(result, 0, sizeof(*result));
    snprintf(result->name, sizeof(result->name), "dataset_replay");
    result->rows = dataset->row_count;
    env = make_eval_env(seed);

    while (result->episodes == 0u && result->steps < max_steps && row_index < dataset->row_count) {
        uint8_t action = NOOP;
        if (env.balls_fired != 0 || env.frameskip > 1) {
            const TkmFloatTransition* row = &dataset->rows[row_index++];
            float distance = obs_distance(env.observations, row->obs);
            if (first_obs_distance < 0.0f) {
                first_obs_distance = distance;
            }
            if (first_drift_row == 0xffffffffu && distance > 0.000001f) {
                first_drift_row = row_index - 1u;
                first_drift_distance = distance;
            }
            if (distance > max_obs_distance) {
                max_obs_distance = distance;
            }
            action = row->action;
            env.actions[0] = (float)action;
            result->action_histogram[action]++;
            c_step(&env);
            if (first_reward_mismatch_row == 0xffffffffu &&
                (env.rewards[0] != row->reward ||
                    ((env.terminals[0] != 0.0f ? 1u : 0u) != row->terminal))) {
                first_reward_mismatch_row = row_index - 1u;
            }
            episode_return += env.rewards[0];
            episode_length++;
            result->steps++;
            if (env.terminals[0] != 0.0f) {
                result->episodes = 1u;
                result->return_sum = episode_return;
                result->length_sum = (float)episode_length;
                result->max_return = episode_return;
            }
            continue;
        }
        env.actions[0] = (float)action;
        result->action_histogram[action]++;
        c_step(&env);
        episode_return += env.rewards[0];
        episode_length++;
        result->steps++;
        if (env.terminals[0] != 0.0f) {
            result->episodes = 1u;
            result->return_sum = episode_return;
            result->length_sum = (float)episode_length;
            result->max_return = episode_return;
        }
    }

    if (episode_length > 0u && result->episodes == 0u) {
        result->episodes = 1u;
        result->return_sum = episode_return;
        result->length_sum = (float)episode_length;
        result->max_return = episode_return;
    }
    *out_max_obs_distance = max_obs_distance;
    free_allocated(&env);
    printf("dataset_replay_first_obs_distance=%.9g\n", first_obs_distance);
    printf("dataset_replay_first_drift_row=%u distance=%.9g first_reward_mismatch_row=%u\n",
        first_drift_row, first_drift_distance, first_reward_mismatch_row);
    return TKM_OK;
}

int main(int argc, char** argv) {
    const char* config_path = getenv("TKM_PUFFERLIB_BREAKOUT_CONFIG");
    const char* jsonl_path = argc > 1 ? argv[1] : getenv("TKM_PUFFERLIB_BREAKOUT_JSONL");
    const char* config_dataset_path = "";
    const char* out_dir = argc > 2 ? argv[2] : "/tmp/tkm_breakout_policy_benchmark";
    uint32_t eval_episodes = read_u32_env("TKM_PUFFERLIB_BREAKOUT_EVAL_EPISODES", DEFAULT_EVAL_EPISODES);
    uint32_t max_steps = read_u32_env("TKM_PUFFERLIB_BREAKOUT_MAX_STEPS", DEFAULT_MAX_STEPS);
    uint32_t seed = read_u32_env("TKM_PUFFERLIB_BREAKOUT_EVAL_SEED", 0u);
    static TkmFloatTransitionDataset all;
    static TkmFloatTransitionDataset top1;
    TkmIni token_ini;
    BreakoutPufferlibTokenPolicy token_policy;
    BreakoutPufferlibTokenPolicy token_top1_policy;
    BreakoutPufferlibTokenConfig token_config;
    BreakoutPufferlibTokenReport token_report;
    BreakoutPufferlibTokenReport token_top1_report;
    BenchPolicy policy;
    BenchResult result;
    float replay_max_obs_distance = 0.0f;

    if (!config_path || config_path[0] == '\0') {
        config_path = DEFAULT_TOKEN_CONFIG_PATH;
    }
    if (load_token_config(
            config_path,
            &token_ini,
            &token_config,
            &config_dataset_path,
            &g_configured_frameskip) != TKM_OK) {
        fprintf(stderr, "failed to load token config: %s\n", config_path);
        return 1;
    }
    if (!jsonl_path || jsonl_path[0] == '\0') {
        jsonl_path = config_dataset_path;
    }
    if (!jsonl_path || jsonl_path[0] == '\0') {
        fprintf(stderr, "usage: %s DATASET_JSONL [OUT_DIR]\n", argv[0]);
        return 1;
    }
    if (make_dir_if_needed(out_dir) != TKM_OK) {
        fprintf(stderr, "failed to prepare output dir: %s\n", out_dir);
        return 1;
    }
    if (tkm_float_transition_dataset_load_jsonl(&all, jsonl_path) != TKM_OK ||
        tkm_float_transition_filter_top_return(&all, &top1, 1u) != TKM_OK) {
        fprintf(stderr, "failed to load/filter dataset: %s\n", jsonl_path);
        return 1;
    }

    printf("pufferlib_breakout_policy_benchmark config=%s dataset=%s frameskip=%u eval_episodes=%u max_steps=%u seed=%u\n",
        config_path, jsonl_path, g_configured_frameskip, eval_episodes, max_steps, seed);

    if (eval_dataset_replay(&all, seed, max_steps, &result, &replay_max_obs_distance) != TKM_OK) {
        fprintf(stderr, "dataset_replay failed\n");
        return 1;
    }
    print_result(&result);
    printf("dataset_replay_max_obs_distance=%.9g\n", replay_max_obs_distance);

    memset(&policy, 0, sizeof(policy));
    if (breakout_pufferlib_token_policy_train(
            &token_policy, &all, &token_config, &token_report) != TKM_OK) {
        fprintf(stderr, "token policy training failed\n");
        return 1;
    }
    printf("token_layers layout=%s tokenizer=%s selection=%s input=%s model=%s head=%s\n",
        token_report.sequence_layout_name,
        token_report.observation_tokenizer_name,
        token_report.observation_selection_name,
        token_report.input_feature_name,
        token_report.sequence_model_name,
        token_report.action_head_name);
    printf("token_train rows=%u train_rows=%u heldout_rows=%u heldout_accuracy=%.3f obs_token_accuracy=%.3f action_token_accuracy=%.3f exact=%u backoff=%u prior=%u bins=%u dims=%u history=%u epochs=%u learning_rate=%.5f input_dim=%u hidden=%u pair_features=%u\n",
        token_report.source_rows,
        token_report.train_rows,
        token_report.heldout_rows,
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
        token_policy.token_mlp_hidden_dim,
        (uint32_t)token_config.use_pair_features);
    printf("token_selected_dims=");
    for (uint32_t i = 0u; i < token_policy.selected_dim_count; i++) {
        printf("%s%u", i == 0u ? "" : ",", token_policy.selected_dims[i]);
    }
    printf("\n");

    memset(&policy, 0, sizeof(policy));
    policy.method = BENCH_METHOD_TOKEN;
    policy.dataset = &all;
    policy.token_policy = &token_policy;
    if (eval_policy(token_result_name(&token_report, 0u), &policy, seed, eval_episodes, max_steps, &result) != TKM_OK) {
        fprintf(stderr, "%s failed\n", token_result_name(&token_report, 0u));
        return 1;
    }
    print_result(&result);

    if (breakout_pufferlib_token_policy_train(
            &token_top1_policy, &top1, &token_config, &token_top1_report) != TKM_OK) {
        fprintf(stderr, "token top1 policy training failed\n");
        return 1;
    }
    printf("token_top1_layers layout=%s tokenizer=%s selection=%s input=%s model=%s head=%s\n",
        token_top1_report.sequence_layout_name,
        token_top1_report.observation_tokenizer_name,
        token_top1_report.observation_selection_name,
        token_top1_report.input_feature_name,
        token_top1_report.sequence_model_name,
        token_top1_report.action_head_name);
    printf("token_top1_train rows=%u train_rows=%u heldout_rows=%u heldout_accuracy=%.3f obs_token_accuracy=%.3f action_token_accuracy=%.3f exact=%u backoff=%u prior=%u bins=%u dims=%u history=%u epochs=%u learning_rate=%.5f input_dim=%u hidden=%u pair_features=%u\n",
        token_top1_report.source_rows,
        token_top1_report.train_rows,
        token_top1_report.heldout_rows,
        token_top1_report.heldout_accuracy,
        token_top1_report.obs_token_accuracy,
        token_top1_report.action_token_accuracy,
        token_top1_report.exact_predictions,
        token_top1_report.backoff_predictions,
        token_top1_report.prior_predictions,
        token_top1_report.bin_count,
        token_top1_report.selected_dim_count,
        token_top1_policy.history,
        token_config.epochs,
        token_config.learning_rate,
        token_top1_policy.token_mlp_input_dim,
        token_top1_policy.token_mlp_hidden_dim,
        (uint32_t)token_config.use_pair_features);
    printf("token_top1_selected_dims=");
    for (uint32_t i = 0u; i < token_top1_policy.selected_dim_count; i++) {
        printf("%s%u", i == 0u ? "" : ",", token_top1_policy.selected_dims[i]);
    }
    printf("\n");

    memset(&policy, 0, sizeof(policy));
    policy.method = BENCH_METHOD_TOKEN;
    policy.dataset = &top1;
    policy.token_policy = &token_top1_policy;
    if (eval_policy(token_result_name(&token_top1_report, 1u), &policy, seed, eval_episodes, max_steps, &result) != TKM_OK) {
        fprintf(stderr, "%s failed\n", token_result_name(&token_top1_report, 1u));
        return 1;
    }
    print_result(&result);

    return 0;
}
