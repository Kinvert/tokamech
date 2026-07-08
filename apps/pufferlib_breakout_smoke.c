#include <errno.h>
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

#define DEFAULT_TOKEN_CONFIG_PATH "projects/breakout/pufferlib_token_ngram.ini"
#define SMOKE_EXPORT_ROWS 256u
#define SMOKE_EVAL_STEPS 512u
#define SMOKE_EVAL_EPISODES 2u

typedef struct {
    uint32_t episodes;
    uint32_t steps;
    uint32_t action_histogram[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float return_sum;
    float score_sum;
    float length_sum;
} SmokeEvalStats;

static int make_dir_if_needed(const char* path) {
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return TKM_OK;
    }
    return TKM_ERR;
}

static int join_path(char* out, size_t out_count, const char* dir, const char* name) {
    if (!out || !dir || !name || snprintf(out, out_count, "%s/%s", dir, name) >= (int)out_count) {
        return TKM_ERR;
    }
    return TKM_OK;
}

static int copy_path(char* out, size_t out_count, const char* path) {
    if (!out || !path || path[0] == '\0' || snprintf(out, out_count, "%s", path) >= (int)out_count) {
        return TKM_ERR;
    }
    return TKM_OK;
}

static int load_token_config(const char* path, TkmIni* ini, BreakoutPufferlibTokenConfig* config) {
    if (!path || !ini || !config) {
        return TKM_ERR;
    }
    if (tkm_ini_parse_file(ini, path) != TKM_OK ||
        breakout_pufferlib_token_config_from_ini(ini, config) != TKM_OK) {
        return TKM_ERR;
    }
    return TKM_OK;
}

static Breakout make_pufferlib_breakout_env(void) {
    Breakout env = {
        .client = NULL,
        .frameskip = 1,
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

static int write_transition_row(
    FILE* file,
    const float* obs,
    uint32_t env_index,
    uint32_t episode,
    uint32_t t,
    uint8_t action,
    float reward,
    uint8_t terminal
) {
    if (!file || !obs || action >= TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        return TKM_ERR;
    }

    if (fprintf(file,
            "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"demo_export\","
            "\"env_index\":%u,\"episode\":%u,\"t\":%u,\"obs_dim\":%u,\"obs\":[",
            env_index,
            episode,
            t,
            TKM_PUFFERLIB_BREAKOUT_OBS_DIM
        ) < 0) {
        return TKM_ERR;
    }
    for (uint32_t i = 0; i < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; i++) {
        if (fprintf(file, "%s%.9g", i == 0u ? "" : ",", obs[i]) < 0) {
            return TKM_ERR;
        }
    }
    if (fprintf(file,
            "],\"action\":%u,\"reward\":%.9g,\"terminal\":%u}\n",
            action,
            reward,
            terminal
        ) < 0) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int write_manifest_from_dataset(const char* manifest_path, const TkmFloatTransitionDataset* dataset) {
    TkmFloatEpisodeSummary episodes[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint32_t episode_count = 0;
    float min_return;
    float max_return;
    float return_sum = 0.0f;
    FILE* file;

    if (!manifest_path || !dataset ||
        tkm_float_transition_episode_returns(dataset, episodes, TKM_FLOAT_TRANSITION_MAX_EPISODES, &episode_count) != TKM_OK ||
        episode_count == 0) {
        return TKM_ERR;
    }

    min_return = episodes[0].episode_return;
    max_return = episodes[0].episode_return;
    for (uint32_t i = 0; i < episode_count; i++) {
        if (episodes[i].episode_return < min_return) {
            min_return = episodes[i].episode_return;
        }
        if (episodes[i].episode_return > max_return) {
            max_return = episodes[i].episode_return;
        }
        return_sum += episodes[i].episode_return;
    }

    file = fopen(manifest_path, "wb");
    if (!file) {
        return TKM_ERR;
    }
    if (fprintf(file,
            "{"
            "\"env\":\"pufferlib_breakout\","
            "\"source\":\"demo_export\","
            "\"obs_dim\":%u,"
            "\"action_count\":%u,"
            "\"row_count\":%u,"
            "\"episode_count\":%u,"
            "\"min_return\":%.9g,"
            "\"max_return\":%.9g,"
            "\"mean_return\":%.9g,"
            "\"exporter\":\"tokamech_pufferlib_breakout_smoke\""
            "}\n",
            dataset->obs_dim,
            dataset->action_count,
            dataset->row_count,
            episode_count,
            min_return,
            max_return,
            return_sum / (float)episode_count
        ) < 0 ||
        fclose(file) != 0) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static uint8_t smoke_exploration_action(uint32_t row, uint32_t t) {
    const uint8_t pattern[] = {
        0u, 1u, 2u, 0u,
        2u, 1u, 0u, 1u,
        2u, 2u, 1u, 0u
    };
    return pattern[(row + 3u * t) % (sizeof(pattern) / sizeof(pattern[0]))];
}

static int export_demo_dataset(const char* jsonl_path, const char* manifest_path, uint32_t max_rows) {
    FILE* file;
    Breakout env;
    static TkmFloatTransitionDataset loaded;
    uint32_t episode = 0;
    uint32_t t = 0;

    env = make_pufferlib_breakout_env();
    file = fopen(jsonl_path, "wb");
    if (!file) {
        free_allocated(&env);
        return TKM_ERR;
    }

    for (uint32_t row = 0; row < max_rows; row++) {
        float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
        uint8_t action = smoke_exploration_action(row, t);
        float reward;
        uint8_t terminal;

        memcpy(obs, env.observations, sizeof(obs));
        env.actions[0] = (float)action;
        c_step(&env);
        reward = env.rewards[0];
        terminal = env.terminals[0] != 0.0f ? 1u : 0u;

        if (write_transition_row(file, obs, 0u, episode, t, action, reward, terminal) != TKM_OK) {
            fclose(file);
            free_allocated(&env);
            return TKM_ERR;
        }

        if (terminal) {
            episode++;
            t = 0;
            c_reset(&env);
        } else {
            t++;
        }
    }

    if (fclose(file) != 0) {
        free_allocated(&env);
        return TKM_ERR;
    }

    free_allocated(&env);

    if (tkm_float_transition_dataset_load_jsonl(&loaded, jsonl_path) != TKM_OK ||
        write_manifest_from_dataset(manifest_path, &loaded) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int train_token_from_dataset(
    const char* jsonl_path,
    const char* token_config_path,
    BreakoutPufferlibTokenPolicy* policy,
    BreakoutPufferlibTokenReport* report
) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenConfig config;

    if (!jsonl_path || !token_config_path || !policy || !report) {
        return TKM_ERR;
    }
    if (load_token_config(token_config_path, &ini, &config) != TKM_OK ||
        tkm_float_transition_dataset_load_jsonl(&dataset, jsonl_path) != TKM_OK ||
        breakout_pufferlib_token_policy_train(policy, &dataset, &config, report) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int add_eval_episode(SmokeEvalStats* stats, float episode_return, uint32_t episode_length) {
    if (!stats || episode_length == 0) {
        return TKM_ERR;
    }
    stats->episodes++;
    stats->return_sum += episode_return;
    stats->score_sum += episode_return;
    stats->length_sum += (float)episode_length;
    return TKM_OK;
}

static int eval_token_no_render(BreakoutPufferlibTokenPolicy* policy, SmokeEvalStats* stats) {
    Breakout env;
    float episode_return = 0.0f;
    uint32_t episode_length = 0;

    if (!policy || !stats) {
        return TKM_ERR;
    }

    memset(stats, 0, sizeof(*stats));
    breakout_pufferlib_token_policy_reset(policy);
    env = make_pufferlib_breakout_env();
    while (stats->steps < SMOKE_EVAL_STEPS && stats->episodes < SMOKE_EVAL_EPISODES) {
        uint8_t action = 0;
        if (breakout_pufferlib_token_policy_predict(policy, env.observations, &action) != TKM_OK) {
            free_allocated(&env);
            return TKM_ERR;
        }
        env.actions[0] = (float)action;
        stats->action_histogram[action]++;

        c_step(&env);
        episode_return += env.rewards[0];
        episode_length++;
        stats->steps++;

        if (env.terminals[0] != 0.0f) {
            if (add_eval_episode(stats, episode_return, episode_length) != TKM_OK) {
                free_allocated(&env);
                return TKM_ERR;
            }
            breakout_pufferlib_token_policy_reset(policy);
            episode_return = 0.0f;
            episode_length = 0;
        }
    }

    if (episode_length > 0u && stats->episodes == 0u) {
        if (add_eval_episode(stats, episode_return, episode_length) != TKM_OK) {
            free_allocated(&env);
            return TKM_ERR;
        }
    }

    free_allocated(&env);
    return TKM_OK;
}

int main(int argc, char** argv) {
    const char* out_dir = argc > 1 ? argv[1] : "/tmp/tkm_pufferlib_breakout_smoke";
    const char* input_jsonl = getenv("TKM_PUFFERLIB_BREAKOUT_JSONL");
    const char* input_manifest = getenv("TKM_PUFFERLIB_BREAKOUT_MANIFEST");
    const char* token_config_path = getenv("TKM_PUFFERLIB_BREAKOUT_CONFIG");
    int use_existing_dataset = input_jsonl && input_jsonl[0] != '\0';
    char jsonl_path[512];
    char manifest_path[512];
    BreakoutPufferlibTokenPolicy token_policy;
    BreakoutPufferlibTokenReport token_report;
    SmokeEvalStats eval;
    TkmFloatTransitionManifest manifest;

    if (!token_config_path || token_config_path[0] == '\0') {
        token_config_path = DEFAULT_TOKEN_CONFIG_PATH;
    }
    if (make_dir_if_needed(out_dir) != TKM_OK) {
        fprintf(stderr, "failed to prepare smoke output path\n");
        return 1;
    }

    if (use_existing_dataset) {
        if (copy_path(jsonl_path, sizeof(jsonl_path), input_jsonl) != TKM_OK ||
            copy_path(manifest_path, sizeof(manifest_path),
                input_manifest && input_manifest[0] != '\0' ? input_manifest : "") != TKM_OK) {
            fprintf(stderr, "failed to read existing smoke dataset paths\n");
            return 1;
        }
    } else if (
        join_path(jsonl_path, sizeof(jsonl_path), out_dir, "transitions-000000.jsonl") != TKM_OK ||
        join_path(manifest_path, sizeof(manifest_path), out_dir, "manifest.json") != TKM_OK ||
        export_demo_dataset(jsonl_path, manifest_path, SMOKE_EXPORT_ROWS) != TKM_OK) {
        fprintf(stderr, "failed to export demo smoke dataset\n");
        return 1;
    }

    if (tkm_float_transition_manifest_load(&manifest, manifest_path) != TKM_OK ||
        train_token_from_dataset(jsonl_path, token_config_path, &token_policy, &token_report) != TKM_OK ||
        eval_token_no_render(&token_policy, &eval) != TKM_OK) {
        fprintf(stderr, "pufferlib breakout smoke failed\n");
        return 1;
    }

    printf("pufferlib breakout smoke\n");
    printf("dataset: %s\n", jsonl_path);
    printf("manifest: %s\n", manifest_path);
    printf("rows: %u episodes: %u return[min=%.3f max=%.3f mean=%.3f]\n",
        manifest.row_count,
        manifest.episode_count,
        manifest.min_return,
        manifest.max_return,
        manifest.mean_return
    );
    printf("token_config: %s\n", token_config_path);
    printf("token_layers layout=%s tokenizer=%s selection=%s input=%s model=%s head=%s\n",
        token_report.sequence_layout_name,
        token_report.observation_tokenizer_name,
        token_report.observation_selection_name,
        token_report.input_feature_name,
        token_report.sequence_model_name,
        token_report.action_head_name
    );
    printf("token_train source_rows=%u heldout_action_accuracy=%.3f obs_token_accuracy=%.3f action_token_accuracy=%.3f exact=%u backoff=%u prior=%u bins=%u dims=%u history=%u\n",
        token_report.source_rows,
        token_report.heldout_accuracy,
        token_report.obs_token_accuracy,
        token_report.action_token_accuracy,
        token_report.exact_predictions,
        token_report.backoff_predictions,
        token_report.prior_predictions,
        token_report.bin_count,
        token_report.selected_dim_count,
        token_policy.history
    );
    printf("eval episodes: %u mean_score: %.3f mean_return: %.3f mean_length: %.3f action_histogram: [%u,%u,%u]\n",
        eval.episodes,
        eval.episodes > 0u ? eval.score_sum / (float)eval.episodes : 0.0f,
        eval.episodes > 0u ? eval.return_sum / (float)eval.episodes : 0.0f,
        eval.episodes > 0u ? eval.length_sum / (float)eval.episodes : 0.0f,
        eval.action_histogram[0],
        eval.action_histogram[1],
        eval.action_histogram[2]
    );

    return 0;
}
