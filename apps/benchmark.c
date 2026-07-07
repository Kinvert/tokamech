#include "apps/benchmark.h"

#include <stdio.h>

#include "core/config/ini.h"
#include "core/config/layer_factory.h"
#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/head/action_mask.h"
#include "core/head/categorical.h"
#include "core/model/linear_policy.h"
#include "core/model/action_scorer.h"
#include "core/config/layer_config.h"
#include "core/config/project_config.h"
#include "core/config/run_config.h"
#include "core/config/train_config.h"
#include "core/train/action_scorer_trainer.h"
#include "core/model/mlp_window.h"
#include "core/model/nearest_policy.h"
#include "core/model/ngram.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "core/train/sparse_lookup_policy.h"
#include "projects/breakout/breakout.h"
#include "projects/centerline/centerline.h"
#include "projects/snake/snake.h"

#define TKM_BENCH_NGRAM_ACTION_OFFSET 192u
#define TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN 8u
#define TKM_BENCH_SNAKE_MLP_HIDDEN 32u
#define TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN 64u
#define TKM_BENCH_SNAKE_MLP_EPOCHS 6u
#define TKM_BENCH_SNAKE_DAGGER_ROUNDS 4u
#define TKM_BENCH_SNAKE_MARGIN_LEARNING_RATE 0.01f
#define TKM_BENCH_SNAKE_MASKED_CE_LEARNING_RATE 0.03f
#define TKM_BENCH_SNAKE_DAGGER_LEARNING_RATE 0.005f
#define TKM_BENCH_SNAKE_ACTION_SCORER_MAX_FEATURES SNAKE_ACTION_MAX_FEATURES_PER_ACTION
#define TKM_BENCH_SNAKE_ACTION_SCORER_HIDDEN 32u

static int tkm_bench_config_file_selects_action_scorer(const char* path, TkmProjectKind expected_project) {
    TkmRunConfig config;

    if (!path || tkm_run_config_from_file(path, &config) != TKM_OK) {
        return 0;
    }

    return config.project.project == expected_project &&
        config.layers.model == TKM_MODEL_ACTION_SCORER;
}

static int tkm_bench_one_hot(uint32_t index, uint32_t count, float* out_features) {
    if (!out_features || index >= count || count == 0) {
        return TKM_ERR;
    }
    for (uint32_t i = 0; i < count; i++) {
        out_features[i] = i == index ? 1.0f : 0.0f;
    }
    return TKM_OK;
}

static void tkm_bench_fill_valid_mask(uint8_t* out_mask, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        out_mask[i] = 1;
    }
}

static int tkm_bench_snake_write_action_mask_strategy(
    const SnakeEnv* env,
    TkmMaskStrategyKind strategy,
    uint8_t* out_mask
) {
    if (strategy == TKM_MASK_STRATEGY_IMMEDIATE) {
        return snake_write_action_mask_immediate(env, out_mask, SNAKE_ACTION_COUNT);
    }
    if (strategy == TKM_MASK_STRATEGY_SURVIVAL) {
        return snake_write_action_mask_survival(env, out_mask, SNAKE_ACTION_COUNT);
    }
    return TKM_ERR;
}

static uint32_t tkm_bench_snake_action_feature_dim(TkmActionFeatureKind action_features) {
    if (action_features == TKM_ACTION_FEATURE_BASIC) {
        return SNAKE_ACTION_FEATURES_PER_ACTION;
    }
    if (action_features == TKM_ACTION_FEATURE_SPACE) {
        return SNAKE_ACTION_SPACE_FEATURES_PER_ACTION;
    }
    if (action_features == TKM_ACTION_FEATURE_FOOD_SPACE) {
        return SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION;
    }
    return 0;
}

static int tkm_bench_snake_write_action_features_kind(
    const SnakeEnv* env,
    TkmActionFeatureKind action_features,
    float* out_features
) {
    if (action_features == TKM_ACTION_FEATURE_BASIC) {
        return snake_write_action_features(env, out_features, SNAKE_ACTION_MAX_FEATURE_COUNT);
    }
    if (action_features == TKM_ACTION_FEATURE_SPACE) {
        return snake_write_action_space_features(env, out_features, SNAKE_ACTION_MAX_FEATURE_COUNT);
    }
    if (action_features == TKM_ACTION_FEATURE_FOOD_SPACE) {
        return snake_write_action_food_space_features(env, out_features, SNAKE_ACTION_MAX_FEATURE_COUNT);
    }
    return TKM_ERR;
}

static float tkm_bench_config_learning_rate(const TkmTrainConfig* config, float default_learning_rate) {
    if (config && config->learning_rate > 0.0f) {
        return config->learning_rate;
    }
    return default_learning_rate;
}

static float tkm_bench_absf(float value) {
    return value < 0.0f ? -value : value;
}

static float tkm_bench_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int tkm_bench_train_obs_action_ngram(
    TkmNgramModel* model,
    const TkmTrajectory* trajectory,
    int16_t obs_shift,
    uint8_t action_count
) {
    char ini_text[96];
    TkmIni ini;
    uint16_t sequence[TKM_TRAJECTORY_MAX_STEPS * 2u];
    uint32_t sequence_len = 0;
    uint32_t vocab_size = TKM_BENCH_NGRAM_ACTION_OFFSET + action_count;

    if (!model || !trajectory || trajectory->len == 0 || action_count == 0) {
        return TKM_ERR;
    }

    snprintf(ini_text, sizeof(ini_text), "[model]\nkind = ngram\nvocab_size = %u\n", vocab_size);
    if (tkm_ini_parse(&ini, ini_text) != TKM_OK ||
        tkm_layer_factory_init_ngram(&ini, model) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < trajectory->len; i++) {
        int32_t obs_token = trajectory->obs_i32[i] + obs_shift;
        uint16_t action_token;

        if (obs_token < 0 || obs_token >= (int32_t)TKM_BENCH_NGRAM_ACTION_OFFSET) {
            return TKM_ERR;
        }
        if (trajectory->actions[i] >= action_count) {
            return TKM_ERR;
        }

        action_token = (uint16_t)(TKM_BENCH_NGRAM_ACTION_OFFSET + trajectory->actions[i]);
        sequence[sequence_len++] = (uint16_t)obs_token;
        sequence[sequence_len++] = action_token;
    }

    return tkm_ngram_train_sequence(model, sequence, sequence_len);
}

static uint8_t tkm_bench_predict_ngram_action(
    const TkmNgramModel* model,
    uint16_t obs_token,
    uint8_t action_count,
    uint8_t default_action
) {
    uint16_t next_token = 0;

    if (tkm_ngram_predict_next(model, obs_token, &next_token) != TKM_OK) {
        return default_action;
    }
    if (next_token < TKM_BENCH_NGRAM_ACTION_OFFSET ||
        next_token >= TKM_BENCH_NGRAM_ACTION_OFFSET + action_count) {
        return default_action;
    }

    return (uint8_t)(next_token - TKM_BENCH_NGRAM_ACTION_OFFSET);
}

TkmBenchResult tkm_bench_centerline(void) {
    TkmTrajectory dataset;
    TkmTrajectory rollout;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    TkmDiscreteActionDecoder decoder;
    CenterlineEnv env;
    TkmRuntime runtime;

    tkm_trajectory_init(&dataset);
    tkm_trajectory_init(&rollout);
    (void)tkm_int_bins_init(&tokenizer, -4, 4);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    (void)tkm_lookup_policy_train(&policy, &tokenizer, &dataset, CENTERLINE_ACTION_COUNT);
    (void)tkm_discrete_action_decoder_init(&decoder);
    (void)tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_LEFT, -1);
    (void)tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_STAY, 0);
    (void)tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_RIGHT, 1);
    centerline_env_init(&env, -3, 4, 8);
    tkm_runtime_init(&runtime, &env, centerline_observe_i16, centerline_step_command, &tokenizer, &policy, &decoder);
    (void)tkm_runtime_run(&runtime, 8, &rollout);

    return (TkmBenchResult){
        .name = "centerline",
        .steps = rollout.len,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_ngram(void) {
    TkmTrajectory dataset;
    TkmNgramModel model;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_bench_train_obs_action_ngram(&model, &dataset, 4, CENTERLINE_ACTION_COUNT) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_ngram"};
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;
        uint16_t obs_token;
        uint8_t action;

        (void)centerline_observe_i16(&env, &obs);
        obs_token = (uint16_t)(obs + 4);
        action = tkm_bench_predict_ngram_action(&model, obs_token, CENTERLINE_ACTION_COUNT, CENTERLINE_ACTION_STAY);
        (void)centerline_action_to_command(action, &command);
        (void)centerline_step_command(&env, command, &reward, &terminal);
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_ngram",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_mlp(void) {
    static const char* ini_text =
        "[model]\n"
        "kind = mlp_window\n"
        "input_dim = 1\n"
        "hidden_dim = 3\n"
        "output_dim = 3\n"
        "w0 = centerline_mlp_w0\n"
        "b0 = centerline_mlp_b0\n"
        "w1 = centerline_mlp_w1\n"
        "b1 = centerline_mlp_b1\n";
    static const char* params_text =
        "centerline_mlp_w0 = 1.0 -1.0 0.0\n"
        "centerline_mlp_b0 = 0.0 0.0 1.0\n"
        "centerline_mlp_w1 = 2.0 0.0 0.0  0.0 0.0 2.0  0.0 1.0 0.0\n"
        "centerline_mlp_b1 = 0.0 0.0 0.0\n";
    TkmIni ini;
    TkmParams params;
    TkmMlpWindow mlp;
    CenterlineEnv env;
    uint32_t steps = 0;

    if (tkm_ini_parse(&ini, ini_text) != TKM_OK ||
        tkm_params_parse_text(&params, params_text) != TKM_OK ||
        tkm_layer_factory_init_mlp_window(&ini, &params, &mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_mlp"};
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float input[1];
        float logits[CENTERLINE_ACTION_COUNT];
        uint16_t action_index = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        input[0] = (float)obs;
        if (tkm_mlp_window_forward(&mlp, input, logits, CENTERLINE_ACTION_COUNT) != TKM_OK ||
            tkm_categorical_argmax(logits, CENTERLINE_ACTION_COUNT, &action_index) != TKM_OK ||
            action_index >= CENTERLINE_ACTION_COUNT ||
            centerline_action_to_command((uint8_t)action_index, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_mlp",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_mlp_trained(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    TkmTrajectory dataset;
    TkmMlpWindow mlp;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_mlp_window_init(&mlp, 1, 3, CENTERLINE_ACTION_COUNT, w0, b0, w1, b1) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_mlp_trained"};
    }

    for (uint32_t epoch = 0; epoch < 40; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float input[1] = {(float)dataset.obs_i32[i]};
            if (tkm_mlp_window_train_mse(&mlp, input, dataset.actions[i], 0.05f) != TKM_OK) {
                return (TkmBenchResult){.name = "centerline_mlp_trained"};
            }
        }
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float input[1];
        float logits[CENTERLINE_ACTION_COUNT];
        uint16_t action_index = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        input[0] = (float)obs;
        if (tkm_mlp_window_forward(&mlp, input, logits, CENTERLINE_ACTION_COUNT) != TKM_OK ||
            tkm_categorical_argmax(logits, CENTERLINE_ACTION_COUNT, &action_index) != TKM_OK ||
            centerline_action_to_command((uint8_t)action_index, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_mlp_trained",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_mlp_ce_trained(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    TkmTrajectory dataset;
    TkmMlpWindow mlp;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_mlp_window_init(&mlp, 1, 3, CENTERLINE_ACTION_COUNT, w0, b0, w1, b1) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_mlp_ce_trained"};
    }

    for (uint32_t epoch = 0; epoch < 30; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float input[1] = {(float)dataset.obs_i32[i]};
            if (tkm_mlp_window_train_cross_entropy(&mlp, input, dataset.actions[i], 0.08f) != TKM_OK) {
                return (TkmBenchResult){.name = "centerline_mlp_ce_trained"};
            }
        }
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float input[1];
        float logits[CENTERLINE_ACTION_COUNT];
        uint16_t action_index = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        input[0] = (float)obs;
        if (tkm_mlp_window_forward(&mlp, input, logits, CENTERLINE_ACTION_COUNT) != TKM_OK ||
            tkm_categorical_argmax(logits, CENTERLINE_ACTION_COUNT, &action_index) != TKM_OK ||
            centerline_action_to_command((uint8_t)action_index, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_mlp_ce_trained",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_mlp_masked_ce_trained(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    TkmTrajectory dataset;
    TkmMlpWindow mlp;
    CenterlineEnv env;
    uint8_t valid_mask[CENTERLINE_ACTION_COUNT];
    uint32_t steps = 0;

    tkm_bench_fill_valid_mask(valid_mask, CENTERLINE_ACTION_COUNT);
    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_mlp_window_init(&mlp, 1, 3, CENTERLINE_ACTION_COUNT, w0, b0, w1, b1) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_mlp_masked_ce_trained"};
    }

    for (uint32_t epoch = 0; epoch < 30; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float input[1] = {(float)dataset.obs_i32[i]};
            if (tkm_mlp_window_train_masked_cross_entropy(&mlp, input, valid_mask, dataset.actions[i], 0.08f) != TKM_OK) {
                return (TkmBenchResult){.name = "centerline_mlp_masked_ce_trained"};
            }
        }
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float input[1];
        float logits[CENTERLINE_ACTION_COUNT];
        uint16_t action_index = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        input[0] = (float)obs;
        if (tkm_mlp_window_forward(&mlp, input, logits, CENTERLINE_ACTION_COUNT) != TKM_OK ||
            tkm_action_mask_argmax(logits, valid_mask, CENTERLINE_ACTION_COUNT, &action_index) != TKM_OK ||
            centerline_action_to_command((uint8_t)action_index, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_mlp_masked_ce_trained",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_sparse_lookup(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_sparse_lookup_policy_train(&policy, &dataset, CENTERLINE_ACTION_COUNT, CENTERLINE_ACTION_STAY) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_sparse_lookup"};
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        uint8_t action;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        action = tkm_sparse_lookup_policy_predict(&policy, (int32_t)obs);
        if (centerline_action_to_command(action, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_sparse_lookup",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_nearest(void) {
    TkmTrajectory dataset;
    TkmNearestPolicy policy;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_nearest_policy_init(&policy, 1, CENTERLINE_ACTION_COUNT, CENTERLINE_ACTION_STAY) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_nearest"};
    }
    for (uint32_t i = 0; i < dataset.len; i++) {
        float feature[1] = {(float)dataset.obs_i32[i]};
        if (tkm_nearest_policy_add(&policy, feature, dataset.actions[i]) != TKM_OK) {
            return (TkmBenchResult){.name = "centerline_nearest"};
        }
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float feature[1];
        uint8_t action = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        feature[0] = (float)obs;
        if (tkm_nearest_policy_predict(&policy, feature, &action) != TKM_OK ||
            centerline_action_to_command(action, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_nearest",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_linear_policy(void) {
    TkmTrajectory dataset;
    TkmLinearPolicy policy;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_oracle(&cfg, &dataset);
    }
    if (tkm_linear_policy_init(&policy, 9, CENTERLINE_ACTION_COUNT) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_linear_policy"};
    }

    for (uint32_t epoch = 0; epoch < 8; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float features[9];
            uint32_t token = (uint32_t)(dataset.obs_i32[i] + 4);
            if (tkm_bench_one_hot(token, 9, features) != TKM_OK ||
                tkm_linear_policy_train_perceptron(&policy, features, dataset.actions[i], 0.25f) != TKM_OK) {
                return (TkmBenchResult){.name = "centerline_linear_policy"};
            }
        }
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float features[9];
        uint8_t action = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK ||
            tkm_bench_one_hot((uint32_t)(obs + 4), 9, features) != TKM_OK ||
            tkm_linear_policy_predict(&policy, features, &action) != TKM_OK ||
            centerline_action_to_command(action, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_linear_policy",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_action_scorer(void) {
    const float w0[2] = {0.0f, 1.0f};
    const float b0[1] = {0.0f};
    const float w1[1] = {1.0f};
    const float b1[1] = {0.0f};
    const uint8_t valid_mask[CENTERLINE_ACTION_COUNT] = {1, 1, 1};
    TkmActionScorer scorer;
    CenterlineEnv env;
    uint32_t steps = 0;

    if (tkm_action_scorer_init(&scorer, 1, 1, 1, w0, b0, w1, b1) != TKM_OK) {
        return (TkmBenchResult){.name = "centerline_action_scorer"};
    }

    centerline_env_init(&env, -3, 4, 8);
    while (steps < 8 && !env.terminal) {
        int16_t obs = 0;
        float state[1];
        float action_features[CENTERLINE_ACTION_COUNT];
        uint16_t action_index = CENTERLINE_ACTION_STAY;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            break;
        }
        state[0] = (float)obs / (float)env.limit;
        for (uint32_t action = 0; action < CENTERLINE_ACTION_COUNT; action++) {
            int8_t candidate_command = 0;
            int16_t next_offset;
            if (centerline_action_to_command((uint8_t)action, &candidate_command) != TKM_OK) {
                return (TkmBenchResult){.name = "centerline_action_scorer"};
            }
            next_offset = (int16_t)(obs + candidate_command);
            action_features[action] = 1.0f - (float)(next_offset < 0 ? -next_offset : next_offset) /
                (float)(env.limit + 1);
        }

        if (tkm_action_scorer_masked_argmax(
                &scorer,
                state,
                action_features,
                valid_mask,
                CENTERLINE_ACTION_COUNT,
                &action_index
            ) != TKM_OK ||
            centerline_action_to_command((uint8_t)action_index, &command) != TKM_OK ||
            centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "centerline_action_scorer",
        .steps = steps,
        .score = env.offset == 0 ? 1 : 0,
        .metric = env.offset == 0 ? 1u : 0u,
    };
}

TkmBenchResult tkm_bench_centerline_action_scorer_config_file(const char* path) {
    TkmBenchResult result;

    if (!tkm_bench_config_file_selects_action_scorer(path, TKM_PROJECT_CENTERLINE)) {
        return (TkmBenchResult){.name = "centerline_action_scorer_config_file"};
    }

    result = tkm_bench_centerline_action_scorer();
    result.name = "centerline_action_scorer_config_file";
    return result;
}

TkmBenchResult tkm_bench_breakout(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    BreakoutEnv env;
    BreakoutRolloutStats stats;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    (void)breakout_policy_tokenizer_init(&tokenizer);
    (void)tkm_lookup_policy_train(&policy, &tokenizer, &dataset, BREAKOUT_ACTION_COUNT);
    breakout_env_init_default(&env);
    breakout_reset(&env);
    (void)breakout_run_token_policy(&env, &policy, 220, &stats);

    return (TkmBenchResult){
        .name = "breakout",
        .steps = stats.steps,
        .score = stats.score,
        .metric = stats.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_ngram(void) {
    TkmTrajectory dataset;
    TkmNgramModel model;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_bench_train_obs_action_ngram(&model, &dataset, 0, BREAKOUT_ACTION_COUNT) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_ngram"};
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        uint16_t obs_token = breakout_tokenize_observation(env.observations);
        uint8_t action = tkm_bench_predict_ngram_action(&model, obs_token, BREAKOUT_ACTION_COUNT, BREAKOUT_NOOP);
        float reward = 0.0f;
        uint8_t terminal = 0;

        (void)breakout_step_env(&env, action, &reward, &terminal);
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_ngram",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_sparse_lookup(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_sparse_lookup_policy_train(&policy, &dataset, BREAKOUT_ACTION_COUNT, BREAKOUT_NOOP) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_sparse_lookup"};
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        int32_t token = (int32_t)breakout_tokenize_observation(env.observations);
        uint8_t action = tkm_sparse_lookup_policy_predict(&policy, token);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (breakout_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_sparse_lookup",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_nearest(void) {
    TkmTrajectory dataset;
    TkmNearestPolicy policy;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_nearest_policy_init(&policy, 1, BREAKOUT_ACTION_COUNT, BREAKOUT_NOOP) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_nearest"};
    }
    for (uint32_t i = 0; i < dataset.len; i++) {
        float feature[1] = {(float)dataset.obs_i32[i]};
        if (tkm_nearest_policy_add(&policy, feature, dataset.actions[i]) != TKM_OK) {
            return (TkmBenchResult){.name = "breakout_nearest"};
        }
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        float feature[1] = {(float)breakout_tokenize_observation(env.observations)};
        uint8_t action = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_nearest_policy_predict(&policy, feature, &action) != TKM_OK ||
            breakout_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_nearest",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_linear_policy(void) {
    TkmTrajectory dataset;
    TkmLinearPolicy policy;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_linear_policy_init(&policy, 5, BREAKOUT_ACTION_COUNT) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_linear_policy"};
    }

    for (uint32_t epoch = 0; epoch < 8; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float features[5];
            if (tkm_bench_one_hot((uint32_t)dataset.obs_i32[i], 5, features) != TKM_OK ||
                tkm_linear_policy_train_perceptron(&policy, features, dataset.actions[i], 0.25f) != TKM_OK) {
                return (TkmBenchResult){.name = "breakout_linear_policy"};
            }
        }
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        float features[5];
        uint8_t action = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_one_hot((uint32_t)breakout_tokenize_observation(env.observations), 5, features) != TKM_OK ||
            tkm_linear_policy_predict(&policy, features, &action) != TKM_OK ||
            breakout_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_linear_policy",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_action_scorer(void) {
    const float w0[2] = {0.0f, 1.0f};
    const float b0[1] = {0.0f};
    const float w1[1] = {1.0f};
    const float b1[1] = {0.0f};
    const uint8_t valid_mask[BREAKOUT_ACTION_COUNT] = {1, 1, 1};
    TkmActionScorer scorer;
    BreakoutEnv env;
    uint32_t steps = 0;

    if (tkm_action_scorer_init(&scorer, 1, 1, 1, w0, b0, w1, b1) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_action_scorer"};
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        float state[1] = {breakout_tokenize_observation(env.observations) / 4.0f};
        float action_features[BREAKOUT_ACTION_COUNT];
        float ball_center = env.ball_x + (float)env.ball_width * 0.5f;
        uint16_t action_index = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        for (uint32_t action = 0; action < BREAKOUT_ACTION_COUNT; action++) {
            float command = 0.0f;
            float next_paddle_x;
            float next_paddle_center;
            float distance;
            if (action == BREAKOUT_LEFT) {
                command = -1.0f;
            } else if (action == BREAKOUT_RIGHT) {
                command = 1.0f;
            }
            next_paddle_x = tkm_bench_clampf(
                env.paddle_x + command * env.paddle_speed * (1.0f / 60.0f),
                0.0f,
                (float)env.width - env.paddle_width
            );
            next_paddle_center = next_paddle_x + env.paddle_width * 0.5f;
            distance = tkm_bench_absf(ball_center - next_paddle_center) / (float)env.width;
            action_features[action] = 1.0f - distance;
        }

        if (tkm_action_scorer_masked_argmax(
                &scorer,
                state,
                action_features,
                valid_mask,
                BREAKOUT_ACTION_COUNT,
                &action_index
            ) != TKM_OK ||
            breakout_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_action_scorer",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_action_scorer_config_file(const char* path) {
    TkmBenchResult result;

    if (!tkm_bench_config_file_selects_action_scorer(path, TKM_PROJECT_BREAKOUT)) {
        return (TkmBenchResult){.name = "breakout_action_scorer_config_file"};
    }

    result = tkm_bench_breakout_action_scorer();
    result.name = "breakout_action_scorer_config_file";
    return result;
}

static int tkm_bench_breakout_mlp_ce_init(TkmMlpWindow* mlp) {
    float w0[5 * TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN];
    float b0[TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN];
    float w1[TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN * BREAKOUT_ACTION_COUNT];
    float b1[BREAKOUT_ACTION_COUNT];

    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t h = 0; h < TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN; h++) {
            int pattern = (int)((i * 19u + h * 7u + 5u) % 13u) - 6;
            w0[i * TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN + h] = (float)pattern * 0.02f;
        }
    }

    for (uint32_t h = 0; h < TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN; h++) {
        b0[h] = 0.02f + (float)(h % 3u) * 0.003f;
        for (uint32_t o = 0; o < BREAKOUT_ACTION_COUNT; o++) {
            int pattern = (int)((h * 11u + o * 5u + 2u) % 9u) - 4;
            w1[h * BREAKOUT_ACTION_COUNT + o] = (float)pattern * 0.01f;
        }
    }

    for (uint32_t o = 0; o < BREAKOUT_ACTION_COUNT; o++) {
        b1[o] = 0.0f;
    }

    return tkm_mlp_window_init(
        mlp,
        5,
        TKM_BENCH_BREAKOUT_MLP_CE_HIDDEN,
        BREAKOUT_ACTION_COUNT,
        w0,
        b0,
        w1,
        b1
    );
}

TkmBenchResult tkm_bench_breakout_mlp_ce_trained(void) {
    TkmTrajectory dataset;
    TkmMlpWindow mlp;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_bench_breakout_mlp_ce_init(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_mlp_ce_trained"};
    }

    for (uint32_t epoch = 0; epoch < 12; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float features[5];
            if (tkm_bench_one_hot((uint32_t)dataset.obs_i32[i], 5, features) != TKM_OK ||
                tkm_mlp_window_train_cross_entropy(&mlp, features, dataset.actions[i], 0.08f) != TKM_OK) {
                return (TkmBenchResult){.name = "breakout_mlp_ce_trained"};
            }
        }
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        float features[5];
        float logits[BREAKOUT_ACTION_COUNT];
        uint16_t action_index = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_one_hot((uint32_t)breakout_tokenize_observation(env.observations), 5, features) != TKM_OK ||
            tkm_mlp_window_forward(&mlp, features, logits, BREAKOUT_ACTION_COUNT) != TKM_OK ||
            tkm_categorical_argmax(logits, BREAKOUT_ACTION_COUNT, &action_index) != TKM_OK ||
            breakout_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_mlp_ce_trained",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_mlp_masked_ce_trained(void) {
    TkmTrajectory dataset;
    TkmMlpWindow mlp;
    BreakoutEnv env;
    uint8_t valid_mask[BREAKOUT_ACTION_COUNT];
    uint32_t steps = 0;

    tkm_bench_fill_valid_mask(valid_mask, BREAKOUT_ACTION_COUNT);
    tkm_trajectory_init(&dataset);
    (void)breakout_collect_oracle_tokens(&dataset);
    if (tkm_bench_breakout_mlp_ce_init(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_mlp_masked_ce_trained"};
    }

    for (uint32_t epoch = 0; epoch < 12; epoch++) {
        for (uint32_t i = 0; i < dataset.len; i++) {
            float features[5];
            if (tkm_bench_one_hot((uint32_t)dataset.obs_i32[i], 5, features) != TKM_OK ||
                tkm_mlp_window_train_masked_cross_entropy(&mlp, features, valid_mask, dataset.actions[i], 0.08f) != TKM_OK) {
                return (TkmBenchResult){.name = "breakout_mlp_masked_ce_trained"};
            }
        }
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        float features[5];
        float logits[BREAKOUT_ACTION_COUNT];
        uint16_t action_index = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_one_hot((uint32_t)breakout_tokenize_observation(env.observations), 5, features) != TKM_OK ||
            tkm_mlp_window_forward(&mlp, features, logits, BREAKOUT_ACTION_COUNT) != TKM_OK ||
            tkm_action_mask_argmax(logits, valid_mask, BREAKOUT_ACTION_COUNT, &action_index) != TKM_OK ||
            breakout_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            break;
        }
        steps++;
    }

    return (TkmBenchResult){
        .name = "breakout_mlp_masked_ce_trained",
        .steps = steps,
        .score = env.score,
        .metric = (uint32_t)env.paddle_hits,
    };
}

TkmBenchResult tkm_bench_snake_lookup(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;

    tkm_trajectory_init(&dataset);
    (void)snake_collect_oracle_tokens(&dataset);
    (void)snake_policy_tokenizer_init(&tokenizer);
    (void)tkm_lookup_policy_train(&policy, &tokenizer, &dataset, SNAKE_ACTION_COUNT);
    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    (void)snake_run_token_policy(&env, &policy, 64, &stats);

    return (TkmBenchResult){
        .name = "snake_lookup",
        .steps = stats.steps,
        .score = stats.score,
        .metric = stats.food_eaten,
    };
}

TkmBenchResult tkm_bench_snake_ngram(void) {
    TkmTrajectory dataset;
    TkmNgramModel model;
    SnakeEnv env;
    uint32_t steps = 0;
    uint32_t food_eaten = 0;
    uint8_t terminal = 0;

    tkm_trajectory_init(&dataset);
    (void)snake_collect_oracle_tokens(&dataset);
    if (tkm_bench_train_obs_action_ngram(&model, &dataset, 0, SNAKE_ACTION_COUNT) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_ngram"};
    }

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    while (steps < 16 && !env.terminal) {
        uint16_t obs_token = snake_tokenize_observation(&env);
        uint8_t action = tkm_bench_predict_ngram_action(&model, obs_token, SNAKE_ACTION_COUNT, SNAKE_RIGHT);
        float reward = 0.0f;

        (void)snake_step_env(&env, action, &reward, &terminal);
        steps++;
        if (reward > 0.0f) {
            food_eaten++;
            break;
        }
    }

    return (TkmBenchResult){
        .name = "snake_ngram",
        .steps = steps,
        .score = env.score,
        .metric = food_eaten,
    };
}

TkmBenchResult tkm_bench_snake_sparse_lookup(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;

    tkm_trajectory_init(&dataset);
    if (snake_collect_planner_tokens(&dataset, 96) != TKM_OK ||
        tkm_sparse_lookup_policy_train(&policy, &dataset, SNAKE_ACTION_COUNT, SNAKE_RIGHT) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_sparse_lookup"};
    }

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    if (snake_run_sparse_token_policy(&env, &policy, 96, &stats) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_sparse_lookup"};
    }

    return (TkmBenchResult){
        .name = "snake_sparse_lookup",
        .steps = stats.steps,
        .score = stats.score,
        .metric = stats.food_eaten,
    };
}

TkmBenchResult tkm_bench_snake_sparse_lookup_multistart(void) {
    const int foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
    };
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    tkm_trajectory_init(&dataset);
    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        if (snake_collect_planner_tokens_for_food(&dataset, foods[i][0], foods[i][1], 96) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_sparse_lookup_multistart"};
        }
    }
    if (tkm_sparse_lookup_policy_train(&policy, &dataset, SNAKE_ACTION_COUNT, SNAKE_RIGHT) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_sparse_lookup_multistart"};
    }

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        SnakeEnv env;
        SnakeRolloutStats stats;

        snake_env_init_default(&env);
        snake_reset_fixed(&env);
        snake_set_food(&env, foods[i][0], foods[i][1]);
        if (snake_run_sparse_token_policy(&env, &policy, 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_sparse_lookup_multistart"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_sparse_lookup_multistart",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static int tkm_bench_snake_nearest_add_planner_rollout(
    TkmNearestPolicy* policy,
    int food_r,
    int food_c,
    uint32_t max_steps
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK) {
            return TKM_ERR;
        }
        action = snake_planner_action(&env);
        if (tkm_nearest_policy_add(policy, features, action) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_run_nearest(
    const TkmNearestPolicy* policy,
    int food_r,
    int food_c,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t action = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            tkm_nearest_policy_predict(policy, features, &action) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

static int tkm_bench_snake_run_nearest_safe(
    const TkmNearestPolicy* policy,
    int food_r,
    int food_c,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t action = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            tkm_nearest_policy_predict(policy, features, &action) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_filter_unsafe_action(&env, action, snake_planner_action(&env));
        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_nearest_multistart(void) {
    const int foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
    };
    TkmNearestPolicy policy;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_nearest_policy_init(&policy, SNAKE_FEATURE_COUNT, SNAKE_ACTION_COUNT, SNAKE_RIGHT) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_nearest_multistart"};
    }

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        if (tkm_bench_snake_nearest_add_planner_rollout(&policy, foods[i][0], foods[i][1], 96) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_nearest_multistart"};
        }
    }

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_nearest(&policy, foods[i][0], foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_nearest_multistart"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_nearest_multistart",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_nearest_safe_holdout(void) {
    const int train_foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
    };
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmNearestPolicy policy;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_nearest_policy_init(&policy, SNAKE_FEATURE_COUNT, SNAKE_ACTION_COUNT, SNAKE_RIGHT) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_nearest_safe_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(train_foods) / sizeof(train_foods[0]); i++) {
        if (tkm_bench_snake_nearest_add_planner_rollout(&policy, train_foods[i][0], train_foods[i][1], 96) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_nearest_safe_holdout"};
        }
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_nearest_safe(&policy, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_nearest_safe_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_nearest_safe_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static uint8_t tkm_bench_snake_is_start_body_cell(int row, int col) {
    return row == 6 && col >= 3 && col <= 5;
}

static uint8_t tkm_bench_snake_is_mlp_holdout_food(int row, int col) {
    return (row == 6 && col == 2) ||
        (row == 3 && col == 8) ||
        (row == 8 && col == 8) ||
        (row == 1 && col == 1) ||
        (row == 1 && col == 8);
}

static int tkm_bench_snake_mlp_init_dims(TkmMlpWindow* mlp, uint32_t input_dim, uint32_t hidden_dim) {
    float w0[SNAKE_ACTION_FEATURE_COUNT * TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN];
    float b0[TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN];
    float w1[TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN * SNAKE_ACTION_COUNT];
    float b1[SNAKE_ACTION_COUNT];

    if (!mlp ||
        input_dim == 0 ||
        input_dim > SNAKE_ACTION_FEATURE_COUNT ||
        hidden_dim == 0 ||
        hidden_dim > TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < input_dim; i++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            int pattern = (int)((i * 37u + h * 17u + 11u) % 17u) - 8;
            w0[i * hidden_dim + h] = (float)pattern * 0.015f;
        }
    }

    for (uint32_t h = 0; h < hidden_dim; h++) {
        b0[h] = 0.02f + (float)(h % 5u) * 0.002f;
        for (uint32_t o = 0; o < SNAKE_ACTION_COUNT; o++) {
            int pattern = (int)((h * 13u + o * 7u + 3u) % 11u) - 5;
            w1[h * SNAKE_ACTION_COUNT + o] = (float)pattern * 0.01f;
        }
    }

    for (uint32_t o = 0; o < SNAKE_ACTION_COUNT; o++) {
        b1[o] = 0.0f;
    }

    return tkm_mlp_window_init(
        mlp,
        input_dim,
        hidden_dim,
        SNAKE_ACTION_COUNT,
        w0,
        b0,
        w1,
        b1
    );
}

static int tkm_bench_snake_mlp_init_hidden(TkmMlpWindow* mlp, uint32_t hidden_dim) {
    return tkm_bench_snake_mlp_init_dims(mlp, SNAKE_FEATURE_COUNT, hidden_dim);
}

static int tkm_bench_snake_mlp_init(TkmMlpWindow* mlp) {
    return tkm_bench_snake_mlp_init_hidden(mlp, TKM_BENCH_SNAKE_MLP_HIDDEN);
}

static int tkm_bench_snake_mlp_train_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (tkm_mlp_window_train_mse(mlp, features, action, learning_rate) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_ce_train_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (tkm_mlp_window_train_cross_entropy(mlp, features, action, learning_rate) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_masked_ce_train_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint8_t action;
        uint8_t has_valid_action = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            snake_write_action_mask(&env, valid_mask, SNAKE_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (!valid_mask[action]) {
            break;
        } else {
            has_valid_action = 1;
        }
        if (!has_valid_action) {
            break;
        }
        if (tkm_mlp_window_train_masked_cross_entropy(mlp, features, valid_mask, action, learning_rate) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_train(TkmMlpWindow* mlp) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_mlp_train_rollout(mlp, r, c, 96, 0.01f) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_ce_train(TkmMlpWindow* mlp) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_mlp_ce_train_rollout(mlp, r, c, 96, 0.03f) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_masked_ce_train(TkmMlpWindow* mlp) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_mlp_masked_ce_train_rollout(mlp, r, c, 96, 0.03f) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_run_mlp_safe(
    const TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        float logits[SNAKE_ACTION_COUNT];
        uint16_t action_index = SNAKE_RIGHT;
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            tkm_mlp_window_forward(mlp, features, logits, SNAKE_ACTION_COUNT) != TKM_OK ||
            tkm_categorical_argmax(logits, SNAKE_ACTION_COUNT, &action_index) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_filter_unsafe_action(&env, (uint8_t)action_index, snake_planner_action(&env));
        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_mlp_safe_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init(&mlp) != TKM_OK ||
        tkm_bench_snake_mlp_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_safe_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_safe(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_safe_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_safe_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_mlp_ce_safe_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init(&mlp) != TKM_OK ||
        tkm_bench_snake_mlp_ce_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_ce_safe_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_safe(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_ce_safe_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_ce_safe_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static int tkm_bench_snake_run_mlp_masked(
    const TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        float logits[SNAKE_ACTION_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t action_index = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            snake_write_action_mask(&env, valid_mask, SNAKE_ACTION_COUNT) != TKM_OK ||
            tkm_mlp_window_forward(mlp, features, logits, SNAKE_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_action_mask_argmax(logits, valid_mask, SNAKE_ACTION_COUNT, &action_index) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_mlp_ce_masked_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init(&mlp) != TKM_OK ||
        tkm_bench_snake_mlp_ce_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_ce_masked_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_masked(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_ce_masked_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_ce_masked_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_mlp_masked_ce_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init(&mlp) != TKM_OK ||
        tkm_bench_snake_mlp_masked_ce_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_masked_ce_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_masked(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_masked_ce_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_masked_ce_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static int tkm_bench_snake_mlp_dagger_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_FEATURE_COUNT];
        float logits[SNAKE_ACTION_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t policy_action = SNAKE_RIGHT;
        uint8_t expert_action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_features(&env, features, SNAKE_FEATURE_COUNT) != TKM_OK ||
            snake_write_action_mask(&env, valid_mask, SNAKE_ACTION_COUNT) != TKM_OK ||
            tkm_mlp_window_forward(mlp, features, logits, SNAKE_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        expert_action = snake_planner_action(&env);
        if (tkm_mlp_window_train_cross_entropy(mlp, features, expert_action, learning_rate) != TKM_OK) {
            return TKM_ERR;
        }

        if (tkm_action_mask_argmax(logits, valid_mask, SNAKE_ACTION_COUNT, &policy_action) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)policy_action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_dagger_train(TkmMlpWindow* mlp) {
    const int train_foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
        {6, 3},
        {3, 3},
        {8, 3},
        {1, 5},
        {10, 5},
    };

    if (tkm_bench_snake_mlp_ce_train(mlp) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t round = 0; round < TKM_BENCH_SNAKE_DAGGER_ROUNDS; round++) {
        for (uint32_t i = 0; i < sizeof(train_foods) / sizeof(train_foods[0]); i++) {
            if (tkm_bench_snake_mlp_dagger_rollout(mlp, train_foods[i][0], train_foods[i][1], 96, 0.005f) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_action_features_ce_train_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_FEATURE_COUNT];
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_action_features(&env, features, SNAKE_ACTION_FEATURE_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (tkm_mlp_window_train_cross_entropy(mlp, features, action, learning_rate) != TKM_OK ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_action_features_ce_train(TkmMlpWindow* mlp) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_mlp_action_features_ce_train_rollout(mlp, r, c, 96, 0.03f) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_action_features_dagger_rollout(
    TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_FEATURE_COUNT];
        float logits[SNAKE_ACTION_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t policy_action = SNAKE_RIGHT;
        uint8_t expert_action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_action_features(&env, features, SNAKE_ACTION_FEATURE_COUNT) != TKM_OK ||
            snake_write_action_mask(&env, valid_mask, SNAKE_ACTION_COUNT) != TKM_OK ||
            tkm_mlp_window_forward(mlp, features, logits, SNAKE_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }

        expert_action = snake_planner_action(&env);
        if (tkm_mlp_window_train_cross_entropy(mlp, features, expert_action, learning_rate) != TKM_OK) {
            return TKM_ERR;
        }

        if (tkm_action_mask_argmax(logits, valid_mask, SNAKE_ACTION_COUNT, &policy_action) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)policy_action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_mlp_action_features_dagger_train(TkmMlpWindow* mlp) {
    const int train_foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
        {6, 3},
        {3, 3},
        {8, 3},
        {1, 5},
        {10, 5},
    };

    if (tkm_bench_snake_mlp_action_features_ce_train(mlp) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t round = 0; round < TKM_BENCH_SNAKE_DAGGER_ROUNDS; round++) {
        for (uint32_t i = 0; i < sizeof(train_foods) / sizeof(train_foods[0]); i++) {
            if (tkm_bench_snake_mlp_action_features_dagger_rollout(
                    mlp,
                    train_foods[i][0],
                    train_foods[i][1],
                    96,
                    0.005f
                ) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_mlp_dagger_masked_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init(&mlp) != TKM_OK ||
        tkm_bench_snake_mlp_dagger_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_dagger_masked_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_masked(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_dagger_masked_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_dagger_masked_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_mlp_dagger64_masked_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init_hidden(&mlp, TKM_BENCH_SNAKE_MLP_WIDE_HIDDEN) != TKM_OK ||
        tkm_bench_snake_mlp_dagger_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_dagger64_masked_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_masked(&mlp, holdout_foods[i][0], holdout_foods[i][1], 96, &stats) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_dagger64_masked_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_dagger64_masked_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static int tkm_bench_snake_run_mlp_action_features_masked(
    const TkmMlpWindow* mlp,
    int food_r,
    int food_c,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_FEATURE_COUNT];
        float logits[SNAKE_ACTION_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t action_index = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_write_action_features(&env, features, SNAKE_ACTION_FEATURE_COUNT) != TKM_OK ||
            snake_write_action_mask(&env, valid_mask, SNAKE_ACTION_COUNT) != TKM_OK ||
            tkm_mlp_window_forward(mlp, features, logits, SNAKE_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_action_mask_argmax(logits, valid_mask, SNAKE_ACTION_COUNT, &action_index) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_mlp_action_features_holdout(void) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmMlpWindow mlp;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_mlp_init_dims(&mlp, SNAKE_ACTION_FEATURE_COUNT, TKM_BENCH_SNAKE_MLP_HIDDEN) != TKM_OK ||
        tkm_bench_snake_mlp_action_features_dagger_train(&mlp) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_mlp_action_features_holdout"};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_mlp_action_features_masked(
                &mlp,
                holdout_foods[i][0],
                holdout_foods[i][1],
                96,
                &stats
            ) != TKM_OK) {
            return (TkmBenchResult){.name = "snake_mlp_action_features_holdout"};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = "snake_mlp_action_features_holdout",
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

static int tkm_bench_snake_action_scorer_init_hidden(
    TkmActionScorer* scorer,
    uint32_t hidden_dim,
    TkmActionFeatureKind action_features
) {
    uint32_t action_dim = tkm_bench_snake_action_feature_dim(action_features);
    const uint32_t input_dim = SNAKE_FEATURE_COUNT + action_dim;
    float w0[(SNAKE_FEATURE_COUNT + TKM_BENCH_SNAKE_ACTION_SCORER_MAX_FEATURES) * TKM_MLP_WINDOW_MAX_HIDDEN];
    float b0[TKM_MLP_WINDOW_MAX_HIDDEN];
    float w1[TKM_MLP_WINDOW_MAX_HIDDEN];
    float b1[1] = {0.0f};
    uint32_t action_base = SNAKE_FEATURE_COUNT;
    uint32_t min_hidden = 6u;

    if (action_dim >= SNAKE_ACTION_SPACE_FEATURES_PER_ACTION) {
        min_hidden = 7u;
    }
    if (action_dim >= SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION) {
        min_hidden = 8u;
    }

    if (!scorer ||
        action_dim == 0u ||
        hidden_dim < min_hidden ||
        hidden_dim > TKM_MLP_WINDOW_MAX_HIDDEN) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < input_dim; i++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            int pattern = (int)((i * 29u + h * 19u + 5u) % 13u) - 6;
            w0[i * hidden_dim + h] = (float)pattern * 0.002f;
        }
    }

    for (uint32_t h = 0; h < hidden_dim; h++) {
        b0[h] = 0.0f;
        w1[h] = 0.0f;
    }

    w0[action_base * hidden_dim + 0u] = 1.0f;
    w1[0] = 0.25f;
    w0[(action_base + 1u) * hidden_dim + 1u] = -1.0f;
    w1[1] = 0.75f;
    w0[(action_base + 2u) * hidden_dim + 2u] = 1.0f;
    w0[(action_base + 2u) * hidden_dim + 3u] = -1.0f;
    w0[(action_base + 3u) * hidden_dim + 4u] = 1.0f;
    w0[(action_base + 3u) * hidden_dim + 5u] = -1.0f;
    w1[2] = -0.05f;
    w1[3] = -0.05f;
    w1[4] = -0.05f;
    w1[5] = -0.05f;
    if (action_dim >= SNAKE_ACTION_SPACE_FEATURES_PER_ACTION) {
        w0[(action_base + 4u) * hidden_dim + 6u] = 1.0f;
        w1[6] = 0.75f;
    }
    if (action_dim >= SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION) {
        w0[(action_base + 5u) * hidden_dim + 7u] = 1.0f;
        w1[7] = 0.75f;
    }

    return tkm_action_scorer_init(
        scorer,
        SNAKE_FEATURE_COUNT,
        action_dim,
        hidden_dim,
        w0,
        b0,
        w1,
        b1
    );
}

static int tkm_bench_snake_action_scorer_train_state(
    TkmActionScorer* scorer,
    const SnakeEnv* env,
    TkmActionFeatureKind action_feature_kind,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    float features[SNAKE_ACTION_MAX_FEATURE_COUNT];
    uint8_t valid_mask[SNAKE_ACTION_COUNT];
    uint8_t expert_action;
    const float* state_features = features;
    const float* action_rows = features + SNAKE_FEATURE_COUNT;

    if (tkm_bench_snake_write_action_features_kind(env, action_feature_kind, features) != TKM_OK ||
        tkm_bench_snake_write_action_mask_strategy(env, mask_strategy, valid_mask) != TKM_OK) {
        return TKM_ERR;
    }

    expert_action = snake_planner_action(env);
    if (!valid_mask[expert_action]) {
        return TKM_OK;
    }

    return tkm_action_scorer_train_configured(
        scorer,
        TKM_TRAIN_LOSS_MARGIN,
        state_features,
        action_rows,
        valid_mask,
        SNAKE_ACTION_COUNT,
        expert_action,
        0.35f,
        learning_rate
    );
}

static int tkm_bench_snake_action_scorer_ce_train_state(
    TkmActionScorer* scorer,
    const SnakeEnv* env,
    TkmActionFeatureKind action_feature_kind,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    float features[SNAKE_ACTION_MAX_FEATURE_COUNT];
    uint8_t valid_mask[SNAKE_ACTION_COUNT];
    uint8_t expert_action;
    const float* state_features = features;
    const float* action_rows = features + SNAKE_FEATURE_COUNT;

    if (tkm_bench_snake_write_action_features_kind(env, action_feature_kind, features) != TKM_OK ||
        tkm_bench_snake_write_action_mask_strategy(env, mask_strategy, valid_mask) != TKM_OK) {
        return TKM_ERR;
    }

    expert_action = snake_planner_action(env);
    if (!valid_mask[expert_action]) {
        return TKM_OK;
    }

    return tkm_action_scorer_train_configured(
        scorer,
        TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY,
        state_features,
        action_rows,
        valid_mask,
        SNAKE_ACTION_COUNT,
        expert_action,
        0.35f,
        learning_rate
    );
}

static int tkm_bench_snake_action_scorer_train_rollout(
    TkmActionScorer* scorer,
    int food_r,
    int food_c,
    uint32_t max_steps,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_snake_action_scorer_train_state(
                scorer,
                &env,
                action_features,
                mask_strategy,
                learning_rate
            ) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_ce_train_rollout(
    TkmActionScorer* scorer,
    int food_r,
    int food_c,
    uint32_t max_steps,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_snake_action_scorer_ce_train_state(
                scorer,
                &env,
                action_features,
                mask_strategy,
                learning_rate
            ) != TKM_OK) {
            return TKM_ERR;
        }

        action = snake_planner_action(&env);
        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_train(
    TkmActionScorer* scorer,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    uint32_t rollout_steps,
    float learning_rate
) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_action_scorer_train_rollout(
                        scorer,
                        r,
                        c,
                        rollout_steps,
                        action_features,
                        mask_strategy,
                        learning_rate
                    ) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_ce_train(
    TkmActionScorer* scorer,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    uint32_t rollout_steps,
    float learning_rate
) {
    for (uint32_t epoch = 0; epoch < TKM_BENCH_SNAKE_MLP_EPOCHS; epoch++) {
        for (int r = 1; r < SNAKE_HEIGHT - 1; r++) {
            for (int c = 1; c < SNAKE_WIDTH - 1; c++) {
                if (tkm_bench_snake_is_start_body_cell(r, c) ||
                    tkm_bench_snake_is_mlp_holdout_food(r, c)) {
                    continue;
                }
                if (tkm_bench_snake_action_scorer_ce_train_rollout(
                        scorer,
                        r,
                        c,
                        rollout_steps,
                        action_features,
                        mask_strategy,
                        learning_rate
                    ) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_dagger_rollout(
    TkmActionScorer* scorer,
    int food_r,
    int food_c,
    uint32_t max_steps,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_MAX_FEATURE_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t policy_action = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_snake_action_scorer_train_state(
                scorer,
                &env,
                action_features,
                mask_strategy,
                learning_rate
            ) != TKM_OK ||
            tkm_bench_snake_write_action_features_kind(&env, action_features, features) != TKM_OK ||
            tkm_bench_snake_write_action_mask_strategy(&env, mask_strategy, valid_mask) != TKM_OK) {
            return TKM_ERR;
        }

        if (tkm_action_scorer_masked_argmax(
                scorer,
                features,
                features + SNAKE_FEATURE_COUNT,
                valid_mask,
                SNAKE_ACTION_COUNT,
                &policy_action
            ) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)policy_action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_ce_dagger_rollout(
    TkmActionScorer* scorer,
    int food_r,
    int food_c,
    uint32_t max_steps,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    float learning_rate
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_MAX_FEATURE_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t policy_action = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_snake_action_scorer_ce_train_state(
                scorer,
                &env,
                action_features,
                mask_strategy,
                learning_rate
            ) != TKM_OK ||
            tkm_bench_snake_write_action_features_kind(&env, action_features, features) != TKM_OK ||
            tkm_bench_snake_write_action_mask_strategy(&env, mask_strategy, valid_mask) != TKM_OK) {
            return TKM_ERR;
        }

        if (tkm_action_scorer_masked_argmax(
                scorer,
                features,
                features + SNAKE_FEATURE_COUNT,
                valid_mask,
                SNAKE_ACTION_COUNT,
                &policy_action
            ) != TKM_OK) {
            break;
        }
        if (snake_step_env(&env, (uint8_t)policy_action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_dagger_train(
    TkmActionScorer* scorer,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    uint32_t rollout_steps,
    uint32_t dagger_rounds,
    float learning_rate,
    float dagger_learning_rate
) {
    const int train_foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
        {6, 3},
        {3, 3},
        {8, 3},
        {1, 5},
        {10, 5},
    };

    if (tkm_bench_snake_action_scorer_train(
            scorer,
            action_features,
            mask_strategy,
            rollout_steps,
            learning_rate
        ) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t round = 0; round < dagger_rounds; round++) {
        for (uint32_t i = 0; i < sizeof(train_foods) / sizeof(train_foods[0]); i++) {
            if (tkm_bench_snake_action_scorer_dagger_rollout(
                    scorer,
                    train_foods[i][0],
                    train_foods[i][1],
                    rollout_steps,
                    action_features,
                    mask_strategy,
                    dagger_learning_rate
                ) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_action_scorer_ce_dagger_train(
    TkmActionScorer* scorer,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    uint32_t rollout_steps,
    uint32_t dagger_rounds,
    float learning_rate,
    float dagger_learning_rate
) {
    const int train_foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
        {6, 3},
        {3, 3},
        {8, 3},
        {1, 5},
        {10, 5},
    };

    if (tkm_bench_snake_action_scorer_ce_train(
            scorer,
            action_features,
            mask_strategy,
            rollout_steps,
            learning_rate
        ) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t round = 0; round < dagger_rounds; round++) {
        for (uint32_t i = 0; i < sizeof(train_foods) / sizeof(train_foods[0]); i++) {
            if (tkm_bench_snake_action_scorer_ce_dagger_rollout(
                    scorer,
                    train_foods[i][0],
                    train_foods[i][1],
                    rollout_steps,
                    action_features,
                    mask_strategy,
                    dagger_learning_rate
                ) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    return TKM_OK;
}

static int tkm_bench_snake_run_action_scorer_masked(
    const TkmActionScorer* scorer,
    int food_r,
    int food_c,
    uint32_t max_steps,
    TkmActionFeatureKind action_features,
    TkmMaskStrategyKind mask_strategy,
    TkmDecoderFallbackKind decoder_fallback,
    uint32_t decoder_rollout_horizon,
    TkmDecoderRolloutScoreKind decoder_rollout_score_mode,
    SnakeRolloutStats* stats
) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);
    *stats = (SnakeRolloutStats){0};

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        float features[SNAKE_ACTION_MAX_FEATURE_COUNT];
        uint8_t valid_mask[SNAKE_ACTION_COUNT];
        uint16_t action_index = SNAKE_RIGHT;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_snake_write_action_features_kind(&env, action_features, features) != TKM_OK ||
            tkm_bench_snake_write_action_mask_strategy(&env, mask_strategy, valid_mask) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_action_scorer_masked_argmax(
                scorer,
                features,
                features + SNAKE_FEATURE_COUNT,
                valid_mask,
                SNAKE_ACTION_COUNT,
                &action_index
            ) != TKM_OK) {
            break;
        }
        if (decoder_fallback == TKM_DECODER_FALLBACK_PLANNER_SCORE) {
            action_index = snake_filter_planner_score_action(&env, (uint8_t)action_index);
        }
        if (decoder_fallback == TKM_DECODER_FALLBACK_BEST_SCORE) {
            action_index = snake_filter_best_score_action(&env);
        }
        if (decoder_fallback == TKM_DECODER_FALLBACK_ROLLOUT_SCORE) {
            action_index = snake_filter_rollout_score_mode_action(
                &env,
                (uint8_t)action_index,
                decoder_rollout_horizon,
                decoder_rollout_score_mode
            );
        }
        if (snake_step_env(&env, (uint8_t)action_index, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env.score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

static TkmBenchResult tkm_bench_snake_action_scorer_holdout_strategy(
    TkmMaskStrategyKind mask_strategy,
    TkmDecoderFallbackKind decoder_fallback,
    uint32_t decoder_rollout_horizon,
    TkmDecoderRolloutScoreKind decoder_rollout_score_mode,
    uint32_t hidden_dim,
    TkmActionFeatureKind action_features,
    uint32_t train_steps,
    uint32_t dagger_rounds,
    float learning_rate,
    float dagger_learning_rate,
    uint32_t eval_steps,
    const char* name
) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmActionScorer scorer;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_action_scorer_init_hidden(&scorer, hidden_dim, action_features) != TKM_OK ||
        tkm_bench_snake_action_scorer_dagger_train(
            &scorer,
            action_features,
            mask_strategy,
            train_steps,
            dagger_rounds,
            learning_rate,
            dagger_learning_rate
        ) != TKM_OK) {
        return (TkmBenchResult){.name = name};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_action_scorer_masked(
                &scorer,
                holdout_foods[i][0],
                holdout_foods[i][1],
                eval_steps,
                action_features,
                mask_strategy,
                decoder_fallback,
                decoder_rollout_horizon,
                decoder_rollout_score_mode,
                &stats
            ) != TKM_OK) {
            return (TkmBenchResult){.name = name};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = name,
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_action_scorer_holdout(void) {
    return tkm_bench_snake_action_scorer_holdout_strategy(
        TKM_MASK_STRATEGY_SURVIVAL,
        TKM_DECODER_FALLBACK_NONE,
        TKM_DECODER_DEFAULT_ROLLOUT_HORIZON,
        TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS,
        TKM_BENCH_SNAKE_ACTION_SCORER_HIDDEN,
        TKM_ACTION_FEATURE_SPACE,
        96,
        TKM_BENCH_SNAKE_DAGGER_ROUNDS,
        TKM_BENCH_SNAKE_MARGIN_LEARNING_RATE,
        TKM_BENCH_SNAKE_DAGGER_LEARNING_RATE,
        96,
        "snake_action_scorer_holdout"
    );
}

static TkmBenchResult tkm_bench_snake_action_scorer_ce_holdout_strategy(
    TkmMaskStrategyKind mask_strategy,
    TkmDecoderFallbackKind decoder_fallback,
    uint32_t decoder_rollout_horizon,
    TkmDecoderRolloutScoreKind decoder_rollout_score_mode,
    uint32_t hidden_dim,
    TkmActionFeatureKind action_features,
    uint32_t train_steps,
    uint32_t dagger_rounds,
    float learning_rate,
    float dagger_learning_rate,
    uint32_t eval_steps,
    const char* name
) {
    const int holdout_foods[][2] = {
        {6, 2},
        {3, 8},
        {8, 8},
        {1, 1},
        {1, 8},
    };
    TkmActionScorer scorer;
    uint32_t total_steps = 0;
    uint32_t total_food = 0;

    if (tkm_bench_snake_action_scorer_init_hidden(&scorer, hidden_dim, action_features) != TKM_OK ||
        tkm_bench_snake_action_scorer_ce_dagger_train(
            &scorer,
            action_features,
            mask_strategy,
            train_steps,
            dagger_rounds,
            learning_rate,
            dagger_learning_rate
        ) != TKM_OK) {
        return (TkmBenchResult){.name = name};
    }

    for (uint32_t i = 0; i < sizeof(holdout_foods) / sizeof(holdout_foods[0]); i++) {
        SnakeRolloutStats stats;
        if (tkm_bench_snake_run_action_scorer_masked(
                &scorer,
                holdout_foods[i][0],
                holdout_foods[i][1],
                eval_steps,
                action_features,
                mask_strategy,
                decoder_fallback,
                decoder_rollout_horizon,
                decoder_rollout_score_mode,
                &stats
            ) != TKM_OK) {
            return (TkmBenchResult){.name = name};
        }
        total_steps += stats.steps;
        total_food += stats.food_eaten;
    }

    return (TkmBenchResult){
        .name = name,
        .steps = total_steps,
        .score = (int)total_food,
        .metric = total_food,
    };
}

TkmBenchResult tkm_bench_snake_action_scorer_ce_holdout(void) {
    return tkm_bench_snake_action_scorer_ce_holdout_strategy(
        TKM_MASK_STRATEGY_SURVIVAL,
        TKM_DECODER_FALLBACK_NONE,
        TKM_DECODER_DEFAULT_ROLLOUT_HORIZON,
        TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS,
        TKM_BENCH_SNAKE_ACTION_SCORER_HIDDEN,
        TKM_ACTION_FEATURE_SPACE,
        96,
        TKM_BENCH_SNAKE_DAGGER_ROUNDS,
        TKM_BENCH_SNAKE_MASKED_CE_LEARNING_RATE,
        TKM_BENCH_SNAKE_DAGGER_LEARNING_RATE,
        96,
        "snake_action_scorer_ce_holdout"
    );
}

TkmBenchResult tkm_bench_snake_action_scorer_config_holdout(const char* train_ini) {
    TkmIni ini;
    TkmRunConfig config;
    TkmBenchResult result;

    if (!train_ini ||
        tkm_ini_parse(&ini, train_ini) != TKM_OK ||
        tkm_run_config_from_ini(&ini, &config) != TKM_OK ||
        config.project.project != TKM_PROJECT_SNAKE ||
        config.layers.model != TKM_MODEL_ACTION_SCORER) {
        return (TkmBenchResult){.name = "snake_action_scorer_config_holdout"};
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MARGIN) {
        result = tkm_bench_snake_action_scorer_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MARGIN_LEARNING_RATE),
            config.train.dagger_learning_rate,
            96,
            "snake_action_scorer_config_margin_holdout"
        );
        result.name = "snake_action_scorer_config_margin_holdout";
        return result;
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY) {
        result = tkm_bench_snake_action_scorer_ce_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MASKED_CE_LEARNING_RATE),
            config.train.dagger_learning_rate,
            96,
            "snake_action_scorer_config_ce_holdout"
        );
        result.name = "snake_action_scorer_config_ce_holdout";
        return result;
    }

    return (TkmBenchResult){.name = "snake_action_scorer_config_holdout"};
}

TkmBenchResult tkm_bench_snake_action_scorer_config_file_holdout(const char* path) {
    TkmRunConfig config;
    TkmBenchResult result;

    if (!path ||
        tkm_run_config_from_file(path, &config) != TKM_OK ||
        config.project.project != TKM_PROJECT_SNAKE ||
        config.layers.model != TKM_MODEL_ACTION_SCORER) {
        return (TkmBenchResult){.name = "snake_action_scorer_config_file_holdout"};
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MARGIN) {
        result = tkm_bench_snake_action_scorer_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MARGIN_LEARNING_RATE),
            config.train.dagger_learning_rate,
            96,
            "snake_action_scorer_config_file_holdout"
        );
        result.name = "snake_action_scorer_config_file_holdout";
        return result;
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY) {
        result = tkm_bench_snake_action_scorer_ce_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MASKED_CE_LEARNING_RATE),
            config.train.dagger_learning_rate,
            96,
            "snake_action_scorer_config_file_holdout"
        );
        result.name = "snake_action_scorer_config_file_holdout";
        return result;
    }

    return (TkmBenchResult){.name = "snake_action_scorer_config_file_holdout"};
}

TkmBenchResult tkm_bench_snake_action_scorer_config_file_long_holdout(const char* path) {
    TkmRunConfig config;
    TkmBenchResult result;

    if (!path ||
        tkm_run_config_from_file(path, &config) != TKM_OK ||
        config.project.project != TKM_PROJECT_SNAKE ||
        config.layers.model != TKM_MODEL_ACTION_SCORER) {
        return (TkmBenchResult){.name = "snake_action_scorer_config_file_long_holdout"};
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MARGIN) {
        result = tkm_bench_snake_action_scorer_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MARGIN_LEARNING_RATE),
            config.train.dagger_learning_rate,
            240,
            "snake_action_scorer_config_file_long_holdout"
        );
        result.name = "snake_action_scorer_config_file_long_holdout";
        return result;
    }

    if (config.train.loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY) {
        result = tkm_bench_snake_action_scorer_ce_holdout_strategy(
            config.mask.strategy,
            config.decoder.fallback,
            config.decoder.rollout_horizon,
            config.decoder.rollout_score_mode,
            config.layers.model_hidden_dim,
            config.layers.action_features,
            config.train.rollout_steps,
            config.train.dagger_rounds,
            tkm_bench_config_learning_rate(&config.train, TKM_BENCH_SNAKE_MASKED_CE_LEARNING_RATE),
            config.train.dagger_learning_rate,
            240,
            "snake_action_scorer_config_file_long_holdout"
        );
        result.name = "snake_action_scorer_config_file_long_holdout";
        return result;
    }

    return (TkmBenchResult){.name = "snake_action_scorer_config_file_long_holdout"};
}

TkmBenchResult tkm_bench_snake_planner(void) {
    SnakeEnv env;
    SnakeRolloutStats stats;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    (void)snake_run_planner_policy(&env, 96, &stats);

    return (TkmBenchResult){
        .name = "snake_planner",
        .steps = stats.steps,
        .score = stats.score,
        .metric = stats.food_eaten,
    };
}

void tkm_print_bench_result(TkmBenchResult result) {
    printf("%s steps=%u score=%d metric=%u\n", result.name, result.steps, result.score, result.metric);
}

#ifndef TKM_BENCHMARK_NO_MAIN
int main(void) {
    tkm_print_bench_result(tkm_bench_centerline());
    tkm_print_bench_result(tkm_bench_centerline_ngram());
    tkm_print_bench_result(tkm_bench_centerline_mlp());
    tkm_print_bench_result(tkm_bench_centerline_mlp_trained());
    tkm_print_bench_result(tkm_bench_centerline_mlp_ce_trained());
    tkm_print_bench_result(tkm_bench_centerline_mlp_masked_ce_trained());
    tkm_print_bench_result(tkm_bench_centerline_sparse_lookup());
    tkm_print_bench_result(tkm_bench_centerline_nearest());
    tkm_print_bench_result(tkm_bench_centerline_linear_policy());
    tkm_print_bench_result(tkm_bench_centerline_action_scorer());
    tkm_print_bench_result(tkm_bench_centerline_action_scorer_config_file("projects/centerline/action_scorer.ini"));
    tkm_print_bench_result(tkm_bench_breakout());
    tkm_print_bench_result(tkm_bench_breakout_ngram());
    tkm_print_bench_result(tkm_bench_breakout_sparse_lookup());
    tkm_print_bench_result(tkm_bench_breakout_nearest());
    tkm_print_bench_result(tkm_bench_breakout_linear_policy());
    tkm_print_bench_result(tkm_bench_breakout_action_scorer());
    tkm_print_bench_result(tkm_bench_breakout_action_scorer_config_file("projects/breakout/action_scorer.ini"));
    tkm_print_bench_result(tkm_bench_breakout_mlp_ce_trained());
    tkm_print_bench_result(tkm_bench_breakout_mlp_masked_ce_trained());
    tkm_print_bench_result(tkm_bench_snake_lookup());
    tkm_print_bench_result(tkm_bench_snake_ngram());
    tkm_print_bench_result(tkm_bench_snake_sparse_lookup());
    tkm_print_bench_result(tkm_bench_snake_sparse_lookup_multistart());
    tkm_print_bench_result(tkm_bench_snake_nearest_multistart());
    tkm_print_bench_result(tkm_bench_snake_nearest_safe_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_safe_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_ce_safe_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_ce_masked_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_masked_ce_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_dagger_masked_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_dagger64_masked_holdout());
    tkm_print_bench_result(tkm_bench_snake_mlp_action_features_holdout());
    tkm_print_bench_result(tkm_bench_snake_action_scorer_holdout());
    tkm_print_bench_result(tkm_bench_snake_action_scorer_ce_holdout());
    tkm_print_bench_result(tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = margin\n[mask]\nstrategy = survival\n"));
    tkm_print_bench_result(tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = masked_cross_entropy\n[mask]\nstrategy = survival\n"));
    tkm_print_bench_result(tkm_bench_snake_action_scorer_config_file_holdout("projects/snake/action_scorer.ini"));
    tkm_print_bench_result(tkm_bench_snake_action_scorer_config_file_long_holdout("projects/snake/action_scorer.ini"));
    tkm_print_bench_result(tkm_bench_snake_planner());
    return 0;
}
#endif
