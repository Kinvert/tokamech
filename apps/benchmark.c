#include "apps/benchmark.h"

#include <stdio.h>
#include <string.h>

#include "core/config/ini.h"
#include "core/config/layer_factory.h"
#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/config/layer_config.h"
#include "core/config/project_config.h"
#include "core/config/run_config.h"
#include "core/config/train_config.h"
#include "core/model/ngram.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "core/train/sparse_lookup_policy.h"
#include "projects/breakout/breakout.h"
#include "projects/centerline/centerline.h"
#include "projects/snake/snake.h"

#define TKM_BENCH_NGRAM_ACTION_OFFSET 1000u
#define TKM_BENCH_BREAKOUT_SCRIPT_STEPS 128u
#define TKM_BENCH_BREAKOUT_POS_BINS 12u
#define TKM_BENCH_BREAKOUT_Y_BINS 8u
#define TKM_BENCH_BREAKOUT_VEL_BINS 3u
#define TKM_BENCH_BREAKOUT_LAUNCH_BINS 2u
#define TKM_BENCH_BREAKOUT_RICH_TOKEN_COUNT \
    (TKM_BENCH_BREAKOUT_POS_BINS * TKM_BENCH_BREAKOUT_POS_BINS * \
     TKM_BENCH_BREAKOUT_Y_BINS * TKM_BENCH_BREAKOUT_VEL_BINS * \
     TKM_BENCH_BREAKOUT_VEL_BINS * TKM_BENCH_BREAKOUT_LAUNCH_BINS)

typedef uint8_t (*TkmBenchBreakoutScript)(uint32_t step);

typedef struct {
    uint16_t rich_tokens[TKM_BENCH_NGRAM_ACTION_OFFSET];
    uint16_t dense_tokens[TKM_BENCH_NGRAM_ACTION_OFFSET];
    uint32_t len;
} TkmBenchTokenMap;


static uint32_t tkm_bench_positive_return_weight(float sampled_return) {
    if (sampled_return <= 0.0f) {
        return 1u;
    }
    if (sampled_return >= 4096.0f) {
        return 4096001u;
    }
    return 1u + (uint32_t)(sampled_return * 1000.0f);
}

static int tkm_bench_train_return_weighted_obs_action_ngram(
    TkmNgramModel* model,
    const TkmTrajectory* trajectory,
    int16_t obs_shift,
    uint8_t action_count,
    float discount
) {
    char ini_text[96];
    TkmIni ini;
    float returns[TKM_TRAJECTORY_MAX_STEPS] = {0.0f};
    uint32_t vocab_size = TKM_BENCH_NGRAM_ACTION_OFFSET + action_count;
    uint32_t start = 0u;

    if (!model || !trajectory || trajectory->len == 0 || action_count == 0 ||
        discount < 0.0f || discount > 1.0f) {
        return TKM_ERR;
    }

    snprintf(ini_text, sizeof(ini_text), "[model]\nkind = ngram\nvocab_size = %u\n", vocab_size);
    if (tkm_ini_parse(&ini, ini_text) != TKM_OK ||
        tkm_layer_factory_init_ngram(&ini, model) != TKM_OK) {
        return TKM_ERR;
    }

    while (start < trajectory->len) {
        uint32_t end = start;
        float running = 0.0f;

        while (end + 1u < trajectory->len && !trajectory->terminals[end]) {
            end++;
        }
        for (uint32_t cursor = end + 1u; cursor > start; cursor--) {
            uint32_t i = cursor - 1u;
            running = trajectory->rewards[i] + discount * running;
            returns[i] = running;
        }
        start = end + 1u;
    }

    for (uint32_t i = 0; i < trajectory->len; i++) {
        int32_t obs_token = trajectory->obs_i32[i] + obs_shift;
        uint16_t action_token;
        uint32_t weight;

        if (obs_token < 0 || obs_token >= (int32_t)TKM_BENCH_NGRAM_ACTION_OFFSET ||
            trajectory->actions[i] >= action_count) {
            return TKM_ERR;
        }

        action_token = (uint16_t)(TKM_BENCH_NGRAM_ACTION_OFFSET + trajectory->actions[i]);
        weight = tkm_bench_positive_return_weight(returns[i]);
        if (tkm_ngram_add_weighted_transition(model, (uint16_t)obs_token, action_token, weight) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
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

static uint32_t tkm_bench_breakout_bucket(float value, float max_value, uint32_t bins) {
    int bucket;

    if (max_value <= 0.0f || bins == 0u) {
        return 0u;
    }

    bucket = (int)((value * (float)bins) / max_value);
    if (bucket < 0) {
        return 0u;
    }
    if ((uint32_t)bucket >= bins) {
        return bins - 1u;
    }
    return (uint32_t)bucket;
}

static uint16_t tkm_bench_breakout_rich_token(const BreakoutEnv* env) {
    uint32_t paddle_bin = tkm_bench_breakout_bucket(
        env->paddle_x + env->paddle_width * 0.5f,
        (float)env->width,
        TKM_BENCH_BREAKOUT_POS_BINS
    );
    uint32_t ball_x_bin = tkm_bench_breakout_bucket(
        env->ball_x + (float)env->ball_width * 0.5f,
        (float)env->width,
        TKM_BENCH_BREAKOUT_POS_BINS
    );
    uint32_t ball_y_bin = tkm_bench_breakout_bucket(
        env->ball_y + (float)env->ball_height * 0.5f,
        (float)env->height,
        TKM_BENCH_BREAKOUT_Y_BINS
    );
    uint32_t vx_bin = env->ball_vx < -0.01f ? 0u : (env->ball_vx > 0.01f ? 2u : 1u);
    uint32_t vy_bin = env->ball_vy < -0.01f ? 0u : (env->ball_vy > 0.01f ? 2u : 1u);
    uint32_t fired_bin = env->balls_fired ? 1u : 0u;

    return (uint16_t)(
        (((((paddle_bin * TKM_BENCH_BREAKOUT_POS_BINS + ball_x_bin) *
            TKM_BENCH_BREAKOUT_Y_BINS + ball_y_bin) *
           TKM_BENCH_BREAKOUT_VEL_BINS + vx_bin) *
          TKM_BENCH_BREAKOUT_VEL_BINS + vy_bin) *
         TKM_BENCH_BREAKOUT_LAUNCH_BINS) +
        fired_bin
    );
}

static void tkm_bench_token_map_init(TkmBenchTokenMap* map) {
    map->len = 0u;
}

static int tkm_bench_token_map_get(
    TkmBenchTokenMap* map,
    uint16_t rich_token,
    uint16_t* dense_token,
    uint8_t insert
) {
    for (uint32_t i = 0; i < map->len; i++) {
        if (map->rich_tokens[i] == rich_token) {
            *dense_token = map->dense_tokens[i];
            return TKM_OK;
        }
    }

    if (!insert || map->len >= TKM_BENCH_NGRAM_ACTION_OFFSET) {
        return TKM_ERR;
    }

    map->rich_tokens[map->len] = rich_token;
    map->dense_tokens[map->len] = (uint16_t)map->len;
    *dense_token = (uint16_t)map->len;
    map->len++;
    return TKM_OK;
}

static uint8_t tkm_bench_breakout_current_script(uint32_t step) {
    return breakout_exploration_action(step);
}

static uint8_t tkm_bench_breakout_right_then_noop_script(uint32_t step) {
    return step < 10u ? BREAKOUT_RIGHT : BREAKOUT_NOOP;
}

static uint8_t tkm_bench_breakout_sweep_script(uint32_t step) {
    return ((step / 12u) % 2u) != 0u ? BREAKOUT_LEFT : BREAKOUT_RIGHT;
}

static uint8_t tkm_bench_breakout_noop_script(uint32_t step) {
    (void)step;
    return BREAKOUT_NOOP;
}

static int tkm_bench_collect_breakout_rich_tokens_with_script(
    TkmTrajectory* trajectory,
    TkmBenchBreakoutScript script,
    uint32_t max_steps,
    TkmBenchTokenMap* token_map
) {
    BreakoutEnv env;
    uint32_t start_len;

    if (!trajectory || !script || max_steps == 0u) {
        return TKM_ERR;
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    start_len = trajectory->len;

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        uint16_t token = tkm_bench_breakout_rich_token(&env);
        uint8_t action = script(i);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (token_map) {
            uint16_t dense_token = 0;
            if (tkm_bench_token_map_get(token_map, token, &dense_token, 1u) != TKM_OK) {
                return TKM_ERR;
            }
            token = dense_token;
        }

        if (breakout_step_env(&env, action, &reward, &terminal) != TKM_OK ||
            tkm_trajectory_append(trajectory, (int32_t)token, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    if (trajectory->len > start_len && !trajectory->terminals[trajectory->len - 1u]) {
        trajectory->terminals[trajectory->len - 1u] = 1u;
    }

    return TKM_OK;
}

static int tkm_bench_collect_breakout_rich_tokens(
    TkmTrajectory* trajectory,
    TkmBenchTokenMap* token_map
) {
    if (tkm_bench_collect_breakout_rich_tokens_with_script(
            trajectory,
            tkm_bench_breakout_current_script,
            TKM_BENCH_BREAKOUT_SCRIPT_STEPS,
            token_map) != TKM_OK ||
        tkm_bench_collect_breakout_rich_tokens_with_script(
            trajectory,
            tkm_bench_breakout_right_then_noop_script,
            TKM_BENCH_BREAKOUT_SCRIPT_STEPS,
            token_map) != TKM_OK ||
        tkm_bench_collect_breakout_rich_tokens_with_script(
            trajectory,
            tkm_bench_breakout_sweep_script,
            TKM_BENCH_BREAKOUT_SCRIPT_STEPS,
            token_map) != TKM_OK ||
        tkm_bench_collect_breakout_rich_tokens_with_script(
            trajectory,
            tkm_bench_breakout_noop_script,
            TKM_BENCH_BREAKOUT_SCRIPT_STEPS,
            token_map) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int tkm_bench_breakout_run_rich_lookup_policy(
    BreakoutEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    BreakoutRolloutStats* stats
) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint16_t token = tkm_bench_breakout_rich_token(env);
        uint8_t action = tkm_lookup_policy_predict(policy, token);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (breakout_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (env->last_hit_paddle) {
            stats->paddle_hits++;
        }
        stats->score = env->score;
        stats->terminal = terminal;
    }

    return TKM_OK;
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
        (void)centerline_collect_exploration(&cfg, &dataset);
        for (uint8_t action = 0; action < CENTERLINE_ACTION_COUNT; action++) {
            CenterlineEnv sweep_env;
            centerline_env_init(&sweep_env, start, 4, 6);
            for (uint32_t i = 0; i < 6 && !sweep_env.terminal; i++) {
                int16_t obs = 0;
                int8_t command = 0;
                float reward = 0.0f;
                uint8_t terminal = 0;

                if (centerline_observe_i16(&sweep_env, &obs) != TKM_OK ||
                    centerline_action_to_command(action, &command) != TKM_OK ||
                    centerline_step_command(&sweep_env, command, &reward, &terminal) != TKM_OK ||
                    tkm_trajectory_append(&dataset, obs, action, reward, terminal) != TKM_OK) {
                    return (TkmBenchResult){.name = "centerline"};
                }
            }
        }
    }
    (void)tkm_lookup_policy_train_return_weighted(&policy, &tokenizer, &dataset, CENTERLINE_ACTION_COUNT, 0.9f);
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
        (void)centerline_collect_exploration(&cfg, &dataset);
        for (uint8_t action = 0; action < CENTERLINE_ACTION_COUNT; action++) {
            CenterlineEnv sweep_env;
            centerline_env_init(&sweep_env, start, 4, 6);
            for (uint32_t i = 0; i < 6 && !sweep_env.terminal; i++) {
                int16_t obs = 0;
                int8_t command = 0;
                float reward = 0.0f;
                uint8_t terminal = 0;

                if (centerline_observe_i16(&sweep_env, &obs) != TKM_OK ||
                    centerline_action_to_command(action, &command) != TKM_OK ||
                    centerline_step_command(&sweep_env, command, &reward, &terminal) != TKM_OK ||
                    tkm_trajectory_append(&dataset, obs, action, reward, terminal) != TKM_OK) {
                    return (TkmBenchResult){.name = "centerline_ngram"};
                }
            }
        }
    }
    if (tkm_bench_train_return_weighted_obs_action_ngram(
            &model, &dataset, 4, CENTERLINE_ACTION_COUNT, 0.9f) != TKM_OK) {
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



TkmBenchResult tkm_bench_centerline_sparse_lookup(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    CenterlineEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    for (int16_t start = -3; start <= 3; start++) {
        CenterlineConfig cfg = {.start_offset = start, .limit = 4, .horizon = 6};
        (void)centerline_collect_exploration(&cfg, &dataset);
        for (uint8_t action = 0; action < CENTERLINE_ACTION_COUNT; action++) {
            CenterlineEnv sweep_env;
            centerline_env_init(&sweep_env, start, 4, 6);
            for (uint32_t i = 0; i < 6 && !sweep_env.terminal; i++) {
                int16_t obs = 0;
                int8_t command = 0;
                float reward = 0.0f;
                uint8_t terminal = 0;

                if (centerline_observe_i16(&sweep_env, &obs) != TKM_OK ||
                    centerline_action_to_command(action, &command) != TKM_OK ||
                    centerline_step_command(&sweep_env, command, &reward, &terminal) != TKM_OK ||
                    tkm_trajectory_append(&dataset, obs, action, reward, terminal) != TKM_OK) {
                    return (TkmBenchResult){.name = "centerline_sparse_lookup"};
                }
            }
        }
    }
    if (tkm_sparse_lookup_policy_train_return_weighted(
            &policy, &dataset, CENTERLINE_ACTION_COUNT, CENTERLINE_ACTION_STAY, 0.9f) != TKM_OK) {
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



TkmBenchResult tkm_bench_breakout(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    BreakoutEnv env;
    BreakoutRolloutStats stats;

    tkm_trajectory_init(&dataset);
    if (tkm_bench_collect_breakout_rich_tokens(&dataset, NULL) != TKM_OK ||
        tkm_int_bins_init(
            &tokenizer,
            0,
            (int32_t)TKM_BENCH_BREAKOUT_RICH_TOKEN_COUNT - 1) != TKM_OK ||
        tkm_lookup_policy_train_return_weighted(
            &policy,
            &tokenizer,
            &dataset,
            BREAKOUT_ACTION_COUNT,
            1.0f) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout"};
    }
    breakout_env_init_default(&env);
    breakout_reset(&env);
    if (tkm_bench_breakout_run_rich_lookup_policy(&env, &policy, 220, &stats) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout"};
    }

    return (TkmBenchResult){
        .name = "breakout",
        .steps = stats.steps,
        .score = stats.score,
        .metric = stats.paddle_hits,
    };
}

TkmBenchResult tkm_bench_breakout_ngram(void) {
    TkmTrajectory dataset;
    TkmBenchTokenMap token_map;
    TkmNgramModel model;
    BreakoutEnv env;
    uint32_t steps = 0;

    tkm_trajectory_init(&dataset);
    tkm_bench_token_map_init(&token_map);
    if (tkm_bench_collect_breakout_rich_tokens(&dataset, &token_map) != TKM_OK ||
        tkm_bench_train_return_weighted_obs_action_ngram(
            &model, &dataset, 0, BREAKOUT_ACTION_COUNT, 1.0f) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_ngram"};
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        uint16_t dense_token = 0;
        uint8_t action = BREAKOUT_NOOP;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (tkm_bench_token_map_get(
                &token_map,
                tkm_bench_breakout_rich_token(&env),
                &dense_token,
                0u) == TKM_OK) {
            action = tkm_bench_predict_ngram_action(
                &model,
                dense_token,
                BREAKOUT_ACTION_COUNT,
                BREAKOUT_NOOP
            );
        }

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
    if (tkm_bench_collect_breakout_rich_tokens(&dataset, NULL) != TKM_OK ||
        tkm_sparse_lookup_policy_train_return_weighted(
            &policy,
            &dataset,
            BREAKOUT_ACTION_COUNT,
            BREAKOUT_NOOP,
            1.0f) != TKM_OK) {
        return (TkmBenchResult){.name = "breakout_sparse_lookup"};
    }

    breakout_env_init_default(&env);
    breakout_reset(&env);
    while (steps < 220 && !env.terminal) {
        int32_t token = (int32_t)tkm_bench_breakout_rich_token(&env);
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





static int tkm_bench_collect_snake_rich_tokens_for_food(
    TkmTrajectory* trajectory,
    int food_r,
    int food_c,
    uint32_t max_steps,
    uint8_t compress_for_ngram
) {
    SnakeEnv env;

    if (!trajectory || max_steps == 0u) {
        return TKM_ERR;
    }

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        uint16_t rich_token = snake_tokenize_rich_observation(&env);
        int32_t token = compress_for_ngram ?
            (int32_t)(rich_token % TKM_BENCH_NGRAM_ACTION_OFFSET) :
            (int32_t)rich_token;
        uint8_t action = snake_exploration_action(i);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (rich_token == TKM_INVALID_TOKEN ||
            snake_step_env(&env, action, &reward, &terminal) != TKM_OK ||
            tkm_trajectory_append(trajectory, token, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

static int tkm_bench_collect_snake_rich_tokens(
    TkmTrajectory* trajectory,
    uint32_t max_steps,
    uint8_t compress_for_ngram
) {
    const int foods[][2] = {
        {6, 8},
        {6, 2},
        {3, 5},
        {8, 5},
        {3, 8},
    };

    if (!trajectory || max_steps == 0u) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        if (tkm_bench_collect_snake_rich_tokens_for_food(
                trajectory,
                foods[i][0],
                foods[i][1],
                max_steps,
                compress_for_ngram) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

TkmBenchResult tkm_bench_snake_lookup(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;

    tkm_trajectory_init(&dataset);
    if (tkm_bench_collect_snake_rich_tokens(&dataset, 96, 0u) != TKM_OK ||
        snake_rich_policy_tokenizer_init(&tokenizer) != TKM_OK ||
        tkm_lookup_policy_train_return_weighted(
            &policy, &tokenizer, &dataset, SNAKE_ACTION_COUNT, 0.9f) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_lookup"};
    }
    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    if (snake_run_rich_token_policy(&env, &policy, 96, &stats) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_lookup"};
    }

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
    if (tkm_bench_collect_snake_rich_tokens_for_food(&dataset, 6, 8, 96, 1u) != TKM_OK ||
        tkm_bench_train_return_weighted_obs_action_ngram(
            &model, &dataset, 0, SNAKE_ACTION_COUNT, 0.9f) != TKM_OK) {
        return (TkmBenchResult){.name = "snake_ngram"};
    }

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    while (steps < 96 && !env.terminal) {
        uint16_t obs_token = (uint16_t)(snake_tokenize_rich_observation(&env) % TKM_BENCH_NGRAM_ACTION_OFFSET);
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
    if (snake_collect_exploration_tokens(&dataset, 96) != TKM_OK ||
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
        if (snake_collect_exploration_tokens_for_food(&dataset, foods[i][0], foods[i][1], 96) != TKM_OK) {
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
















void tkm_print_bench_result(TkmBenchResult result) {
    printf("%s steps=%u score=%d metric=%u\n", result.name, result.steps, result.score, result.metric);
}

#ifndef TKM_BENCHMARK_NO_MAIN
int main(void) {
    tkm_print_bench_result(tkm_bench_centerline());
    tkm_print_bench_result(tkm_bench_centerline_ngram());
    tkm_print_bench_result(tkm_bench_centerline_sparse_lookup());
    tkm_print_bench_result(tkm_bench_breakout());
    tkm_print_bench_result(tkm_bench_breakout_ngram());
    tkm_print_bench_result(tkm_bench_breakout_sparse_lookup());
    tkm_print_bench_result(tkm_bench_snake_lookup());
    tkm_print_bench_result(tkm_bench_snake_ngram());
    tkm_print_bench_result(tkm_bench_snake_sparse_lookup());
    tkm_print_bench_result(tkm_bench_snake_sparse_lookup_multistart());
    return 0;
}
#endif
