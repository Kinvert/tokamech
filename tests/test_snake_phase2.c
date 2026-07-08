#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/dataset/trajectory.h"
#include "core/config/run_config.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/snake/snake.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)


static void test_snake_reset_writes_flat_observation_and_walls(void) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(env.width == 12);
    CHECK(env.height == 12);
    CHECK(env.length == 3);
    CHECK(env.head_r == 6);
    CHECK(env.head_c == 5);
    CHECK(env.food_r == 6);
    CHECK(env.food_c == 8);
    CHECK(env.grid[0] == SNAKE_WALL);
    CHECK(env.grid[6 * env.width + 5] == SNAKE_BODY);
    CHECK(env.grid[6 * env.width + 8] == SNAKE_FOOD);
    CHECK(env.observations[0] == 6);
    CHECK(env.observations[1] == 5);
    CHECK(env.observations[2] == 6);
    CHECK(env.observations[3] == 8);
    CHECK(env.observations[4] == SNAKE_RIGHT);
}

static void test_snake_movement_food_growth_wall_and_self_collision(void) {
    SnakeEnv env;
    float reward = 0.0f;
    uint8_t terminal = 0;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_step_env(&env, SNAKE_RIGHT, &reward, &terminal) == TKM_OK);
    CHECK(env.head_r == 6);
    CHECK(env.head_c == 6);
    CHECK(env.length == 3);
    CHECK(reward == 0.0f);
    CHECK(terminal == 0);

    snake_set_food(&env, 6, 7);
    CHECK(snake_step_env(&env, SNAKE_RIGHT, &reward, &terminal) == TKM_OK);
    CHECK(env.head_c == 7);
    CHECK(env.length == 4);
    CHECK(env.score == 1);
    CHECK(reward > 0.0f);
    CHECK(terminal == 0);

    snake_reset_fixed(&env);
    env.head_r = 1;
    env.head_c = 5;
    env.length = 1;
    env.body_r[0] = 1;
    env.body_c[0] = 5;
    env.dir = SNAKE_UP;
    snake_rebuild_grid(&env);
    CHECK(snake_step_env(&env, SNAKE_UP, &reward, &terminal) == TKM_OK);
    CHECK(terminal == 1);
    CHECK(reward < 0.0f);

    snake_reset_fixed(&env);
    env.length = 4;
    env.head_r = 5;
    env.head_c = 5;
    env.body_r[0] = 5;
    env.body_c[0] = 5;
    env.body_r[1] = 6;
    env.body_c[1] = 5;
    env.body_r[2] = 6;
    env.body_c[2] = 6;
    env.body_r[3] = 5;
    env.body_c[3] = 6;
    env.dir = SNAKE_DOWN;
    snake_rebuild_grid(&env);
    CHECK(snake_step_env(&env, SNAKE_DOWN, &reward, &terminal) == TKM_OK);
    CHECK(terminal == 1);
}

static void test_snake_allows_non_growing_move_into_old_tail(void) {
    SnakeEnv env;
    float reward = 0.0f;
    uint8_t terminal = 0;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    env.length = 4;
    env.head_r = 5;
    env.head_c = 5;
    env.body_r[0] = 5;
    env.body_c[0] = 5;
    env.body_r[1] = 5;
    env.body_c[1] = 4;
    env.body_r[2] = 6;
    env.body_c[2] = 4;
    env.body_r[3] = 6;
    env.body_c[3] = 5;
    env.dir = SNAKE_LEFT;
    env.food_r = 1;
    env.food_c = 1;
    snake_rebuild_grid(&env);

    CHECK(snake_step_env(&env, SNAKE_DOWN, &reward, &terminal) == TKM_OK);
    CHECK(terminal == 0);
    CHECK(reward == 0.0f);
    CHECK(env.length == 4);
    CHECK(env.head_r == 6);
    CHECK(env.head_c == 5);
    CHECK(env.body_r[3] == 6);
    CHECK(env.body_c[3] == 4);
}

static void test_snake_tokenizer_and_exploration_action_are_valid(void) {
    SnakeEnv env;
    uint16_t token;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    token = snake_tokenize_observation(&env);
    CHECK(token < SNAKE_TOKEN_COUNT);
    CHECK(snake_policy_tokenizer_init(&(TkmIntBins){0}) == TKM_OK);
    CHECK(snake_exploration_action(0u) < SNAKE_ACTION_COUNT);
    CHECK(snake_exploration_action(11u) < SNAKE_ACTION_COUNT);
}

static void test_snake_writes_continuous_features(void) {
    SnakeEnv env;
    float features[SNAKE_FEATURE_COUNT];

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_write_features(&env, features, SNAKE_FEATURE_COUNT) == TKM_OK);
    CHECK(features[0] > 0.0f);
    CHECK(features[1] > 0.0f);
    CHECK(features[2] > 0.0f);
    CHECK(features[3] > 0.0f);
    CHECK(features[4] == 0.0f);
    CHECK(features[5] > 0.0f);
    CHECK(features[6] == 0.0f);
    CHECK(features[7] == 0.0f);
    CHECK(features[8] == 0.0f);
    CHECK(features[9] == 1.0f);
    CHECK(features[10] > 0.0f);
    CHECK(features[11] == 0.0f);
    CHECK(features[12 + (6 * SNAKE_WIDTH + 5)] == 1.0f);
    CHECK(features[12 + (6 * SNAKE_WIDTH + 8)] == -1.0f);
    CHECK(snake_write_features(&env, features, SNAKE_FEATURE_COUNT - 1u) == TKM_ERR);
}

static void test_snake_decode_action_does_not_replace_terminal_model_actions(void) {
    SnakeEnv env;
    TkmDecoderConfig decoder = {
        .fallback = TKM_DECODER_FALLBACK_NONE,
        .rollout_horizon = TKM_DECODER_DEFAULT_ROLLOUT_HORIZON,
    };

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    CHECK(snake_decode_action(&env, &decoder, SNAKE_RIGHT) == SNAKE_RIGHT);

    env.head_r = 1;
    env.head_c = 5;
    env.length = 1;
    env.body_r[0] = 1;
    env.body_c[0] = 5;
    env.dir = SNAKE_UP;
    snake_rebuild_grid(&env);
    CHECK(snake_decode_action(&env, &decoder, SNAKE_UP) == SNAKE_UP);
}



static void test_snake_exploration_collection_policy_training_smoke(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;
    int32_t token;

    tkm_trajectory_init(&dataset);
    CHECK(snake_collect_exploration_tokens(&dataset, 32) == TKM_OK);
    CHECK(dataset.len > 0);
    CHECK(tkm_sparse_lookup_policy_train(&policy, &dataset, SNAKE_ACTION_COUNT, SNAKE_RIGHT) == TKM_OK);

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    token = snake_tokenize_full_observation(&env);
    CHECK(token >= 0);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, token) < SNAKE_ACTION_COUNT);

    CHECK(snake_run_sparse_token_policy(&env, &policy, 32, &stats) == TKM_OK);
    CHECK(stats.steps > 0);
    CHECK(stats.score >= 0);
}

static void test_snake_exploration_collection_supports_multiple_foods(void) {
    const int foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
    };
    TkmTrajectory dataset;

    tkm_trajectory_init(&dataset);
    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        CHECK(snake_collect_exploration_tokens_for_food(&dataset, foods[i][0], foods[i][1], 32) == TKM_OK);
    }
    CHECK(dataset.len > 0);
}

int main(void) {
    test_snake_reset_writes_flat_observation_and_walls();
    test_snake_movement_food_growth_wall_and_self_collision();
    test_snake_allows_non_growing_move_into_old_tail();
    test_snake_tokenizer_and_exploration_action_are_valid();
    test_snake_writes_continuous_features();
    test_snake_decode_action_does_not_replace_terminal_model_actions();
    test_snake_exploration_collection_policy_training_smoke();
    test_snake_exploration_collection_supports_multiple_foods();
    puts("snake phase2 tests passed");
    return 0;
}
