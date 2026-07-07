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

extern int snake_run_planner_policy(SnakeEnv* env, uint32_t max_steps, SnakeRolloutStats* stats);

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

static void test_snake_tokenizer_and_oracle_choose_path_to_food(void) {
    SnakeEnv env;
    uint16_t token;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    token = snake_tokenize_observation(&env);
    CHECK(token < SNAKE_TOKEN_COUNT);
    CHECK(snake_policy_tokenizer_init(&(TkmIntBins){0}) == TKM_OK);
    CHECK(snake_oracle_action(&env) == SNAKE_RIGHT);

    snake_set_food(&env, 3, 5);
    CHECK(snake_oracle_action(&env) == SNAKE_UP);

    snake_set_food(&env, 8, 5);
    CHECK(snake_oracle_action(&env) == SNAKE_DOWN);
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

static void test_snake_writes_action_aware_features(void) {
    SnakeEnv env;
    float features[SNAKE_ACTION_FEATURE_COUNT];
    uint32_t base = SNAKE_FEATURE_COUNT;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_write_action_features(&env, features, SNAKE_ACTION_FEATURE_COUNT) == TKM_OK);
    CHECK(SNAKE_ACTION_FEATURE_COUNT == SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * 4u);
    CHECK(features[12 + (6 * SNAKE_WIDTH + 5)] == 1.0f);
    CHECK(features[base + SNAKE_UP * 4u + 0u] == 1.0f);
    CHECK(features[base + SNAKE_DOWN * 4u + 0u] == 1.0f);
    CHECK(features[base + SNAKE_LEFT * 4u + 0u] == 0.0f);
    CHECK(features[base + SNAKE_RIGHT * 4u + 0u] == 1.0f);
    CHECK(features[base + SNAKE_RIGHT * 4u + 1u] < 0.0f);
    CHECK(features[base + SNAKE_LEFT * 4u + 1u] > 0.0f);
    CHECK(snake_write_action_features(&env, features, SNAKE_ACTION_FEATURE_COUNT - 1u) == TKM_ERR);
}

static void test_snake_writes_action_space_features(void) {
    SnakeEnv env;
    float features[SNAKE_ACTION_SPACE_FEATURE_COUNT];
    uint32_t base = SNAKE_FEATURE_COUNT;
    uint32_t stride = SNAKE_ACTION_SPACE_FEATURES_PER_ACTION;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_write_action_space_features(&env, features, SNAKE_ACTION_SPACE_FEATURE_COUNT) == TKM_OK);
    CHECK(SNAKE_ACTION_SPACE_FEATURE_COUNT == SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * 5u);
    CHECK(features[base + SNAKE_RIGHT * stride + 0u] == 1.0f);
    CHECK(features[base + SNAKE_RIGHT * stride + 1u] < 0.0f);
    CHECK(features[base + SNAKE_RIGHT * stride + 4u] > 0.0f);
    CHECK(features[base + SNAKE_LEFT * stride + 0u] == 0.0f);
    CHECK(features[base + SNAKE_LEFT * stride + 4u] == 0.0f);
    CHECK(snake_write_action_space_features(&env, features, SNAKE_ACTION_SPACE_FEATURE_COUNT - 1u) == TKM_ERR);
}

extern int snake_write_action_food_space_features(const SnakeEnv* env, float* out_features, uint32_t max_features);

static void test_snake_writes_action_food_space_features(void) {
    SnakeEnv env;
    float features[SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * 6u];
    uint32_t base = SNAKE_FEATURE_COUNT;
    uint32_t stride = 6u;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, 6, 6);

    CHECK(snake_write_action_food_space_features(&env, features, (uint32_t)(SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * 6u)) == TKM_OK);
    CHECK(features[base + SNAKE_RIGHT * stride + 0u] == 1.0f);
    CHECK(features[base + SNAKE_RIGHT * stride + 1u] < 0.0f);
    CHECK(features[base + SNAKE_RIGHT * stride + 4u] > 0.0f);
    CHECK(features[base + SNAKE_RIGHT * stride + 5u] == 1.0f);
    CHECK(features[base + SNAKE_UP * stride + 5u] == 0.0f);
    CHECK(snake_write_action_food_space_features(&env, features, (uint32_t)(SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * 6u - 1u)) == TKM_ERR);
}

static void test_snake_filter_replaces_immediately_terminal_actions(void) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    CHECK(snake_filter_unsafe_action(&env, SNAKE_RIGHT, SNAKE_UP) == SNAKE_RIGHT);

    env.head_r = 1;
    env.head_c = 5;
    env.length = 1;
    env.body_r[0] = 1;
    env.body_c[0] = 5;
    env.dir = SNAKE_UP;
    snake_rebuild_grid(&env);
    CHECK(snake_filter_unsafe_action(&env, SNAKE_UP, SNAKE_RIGHT) == SNAKE_RIGHT);
}

static void test_snake_writes_action_mask_for_immediate_collisions(void) {
    SnakeEnv env;
    uint8_t mask[SNAKE_ACTION_COUNT];

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    CHECK(snake_write_action_mask(&env, mask, SNAKE_ACTION_COUNT) == TKM_OK);
    CHECK(mask[SNAKE_UP] == 1);
    CHECK(mask[SNAKE_DOWN] == 1);
    CHECK(mask[SNAKE_LEFT] == 0);
    CHECK(mask[SNAKE_RIGHT] == 1);

    env.head_r = 1;
    env.head_c = 5;
    env.length = 1;
    env.body_r[0] = 1;
    env.body_c[0] = 5;
    env.dir = SNAKE_UP;
    snake_rebuild_grid(&env);
    CHECK(snake_write_action_mask(&env, mask, SNAKE_ACTION_COUNT) == TKM_OK);
    CHECK(mask[SNAKE_UP] == 0);
    CHECK(mask[SNAKE_DOWN] == 1);
    CHECK(mask[SNAKE_LEFT] == 1);
    CHECK(mask[SNAKE_RIGHT] == 1);
    CHECK(snake_write_action_mask(&env, mask, SNAKE_ACTION_COUNT - 1u) == TKM_ERR);
}

static void test_snake_action_mask_filters_dead_end_when_safe_action_exists(void) {
    SnakeEnv env;
    uint8_t immediate_mask[SNAKE_ACTION_COUNT];
    uint8_t mask[SNAKE_ACTION_COUNT];

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    env.length = 8;
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
    env.body_r[4] = 6;
    env.body_c[4] = 6;
    env.body_r[5] = 5;
    env.body_c[5] = 7;
    env.body_r[6] = 4;
    env.body_c[6] = 6;
    env.body_r[7] = 4;
    env.body_c[7] = 7;
    env.dir = SNAKE_RIGHT;
    env.food_r = 1;
    env.food_c = 1;
    snake_rebuild_grid(&env);

    CHECK(snake_write_action_mask_immediate(&env, immediate_mask, SNAKE_ACTION_COUNT) == TKM_OK);
    CHECK(snake_write_action_mask_survival(&env, mask, SNAKE_ACTION_COUNT) == TKM_OK);
    CHECK(immediate_mask[SNAKE_RIGHT] == 1);
    CHECK(mask[SNAKE_UP] == 1);
    CHECK(mask[SNAKE_RIGHT] == 0);
    CHECK(snake_write_action_mask(&env, mask, SNAKE_ACTION_COUNT) == TKM_OK);
    CHECK(mask[SNAKE_RIGHT] == 0);
    CHECK(snake_filter_planner_score_action(&env, SNAKE_RIGHT) == SNAKE_UP);
    CHECK(snake_filter_best_score_action(&env) == SNAKE_UP);
    CHECK(snake_filter_rollout_score_action(&env, SNAKE_RIGHT, 8u) == SNAKE_UP);
    CHECK(snake_filter_rollout_score_mode_action(
        &env,
        SNAKE_RIGHT,
        8u,
        TKM_DECODER_ROLLOUT_SCORE_SPACE_DISTANCE
    ) == SNAKE_UP);
}

static void test_snake_oracle_collection_policy_training_and_runtime_success(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;
    uint16_t token;

    tkm_trajectory_init(&dataset);
    CHECK(snake_collect_oracle_tokens(&dataset) == TKM_OK);
    CHECK(dataset.len > 20);
    CHECK(snake_policy_tokenizer_init(&tokenizer) == TKM_OK);
    CHECK(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, SNAKE_ACTION_COUNT) == TKM_OK);

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    token = snake_tokenize_observation(&env);
    CHECK(tkm_lookup_policy_predict(&policy, token) == SNAKE_RIGHT);

    CHECK(snake_run_token_policy(&env, &policy, 16, &stats) == TKM_OK);
    CHECK(stats.steps >= 3);
    CHECK(stats.food_eaten >= 1);
    CHECK(stats.score >= 1);
    CHECK(stats.terminal == 0);
}

static void test_snake_planner_policy_collects_multiple_foods_without_terminal(void) {
    SnakeEnv env;
    SnakeRolloutStats stats;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_run_planner_policy(&env, 96, &stats) == TKM_OK);
    CHECK(stats.steps >= 12);
    CHECK(stats.food_eaten >= 4);
    CHECK(stats.score >= 4);
    CHECK(stats.terminal == 0);
}

static void test_snake_decoder_config_policy_uses_project_ini(void) {
    TkmRunConfig config;
    SnakeEnv env;
    SnakeRolloutStats stats;

    CHECK(tkm_run_config_from_file("projects/snake/action_scorer.ini", &config) == TKM_OK);
    CHECK(config.decoder.fallback == TKM_DECODER_FALLBACK_ROLLOUT_SCORE);
    CHECK(config.decoder.rollout_horizon == 32);

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    CHECK(snake_decode_action(&env, &config.decoder, SNAKE_LEFT) == SNAKE_RIGHT);
    CHECK(snake_run_decoder_config_policy(&env, &config.decoder, 96, &stats) == TKM_OK);
    CHECK(stats.steps == 96);
    CHECK(stats.food_eaten >= 12);
    CHECK(stats.score >= 12);
    CHECK(stats.terminal == 0);
}

static void test_snake_rich_lookup_distills_planner_for_full_rollout(void) {
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    SnakeEnv env;
    SnakeRolloutStats stats;
    int32_t token;

    tkm_trajectory_init(&dataset);
    CHECK(snake_collect_planner_tokens(&dataset, 96) == TKM_OK);
    CHECK(dataset.len >= 90);
    CHECK(tkm_sparse_lookup_policy_train(&policy, &dataset, SNAKE_ACTION_COUNT, SNAKE_RIGHT) == TKM_OK);
    CHECK(policy.item_count >= 90);

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    token = snake_tokenize_full_observation(&env);
    CHECK(token >= 0);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, token) == SNAKE_RIGHT);

    CHECK(snake_run_sparse_token_policy(&env, &policy, 96, &stats) == TKM_OK);
    CHECK(stats.steps >= 90);
    CHECK(stats.food_eaten >= 8);
    CHECK(stats.score >= 8);
    CHECK(stats.terminal == 0);
}

static void test_snake_sparse_lookup_distills_multiple_starting_foods(void) {
    const int foods[][2] = {
        {6, 8},
        {3, 5},
        {8, 5},
    };
    TkmTrajectory dataset;
    TkmSparseLookupPolicy policy;
    uint32_t total_food = 0;

    tkm_trajectory_init(&dataset);
    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        CHECK(snake_collect_planner_tokens_for_food(&dataset, foods[i][0], foods[i][1], 96) == TKM_OK);
    }
    CHECK(dataset.len >= 270);
    CHECK(tkm_sparse_lookup_policy_train(&policy, &dataset, SNAKE_ACTION_COUNT, SNAKE_RIGHT) == TKM_OK);

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        SnakeEnv env;
        SnakeRolloutStats stats;

        snake_env_init_default(&env);
        snake_reset_fixed(&env);
        snake_set_food(&env, foods[i][0], foods[i][1]);
        CHECK(snake_run_sparse_token_policy(&env, &policy, 96, &stats) == TKM_OK);
        CHECK(stats.steps >= 90);
        CHECK(stats.food_eaten >= 6);
        CHECK(stats.terminal == 0);
        total_food += stats.food_eaten;
    }

    CHECK(total_food >= 24);
}

int main(void) {
    test_snake_reset_writes_flat_observation_and_walls();
    test_snake_movement_food_growth_wall_and_self_collision();
    test_snake_allows_non_growing_move_into_old_tail();
    test_snake_tokenizer_and_oracle_choose_path_to_food();
    test_snake_writes_continuous_features();
    test_snake_writes_action_aware_features();
    test_snake_writes_action_space_features();
    test_snake_writes_action_food_space_features();
    test_snake_filter_replaces_immediately_terminal_actions();
    test_snake_writes_action_mask_for_immediate_collisions();
    test_snake_action_mask_filters_dead_end_when_safe_action_exists();
    test_snake_oracle_collection_policy_training_and_runtime_success();
    test_snake_planner_policy_collects_multiple_foods_without_terminal();
    test_snake_decoder_config_policy_uses_project_ini();
    test_snake_rich_lookup_distills_planner_for_full_rollout();
    test_snake_sparse_lookup_distills_multiple_starting_foods();
    puts("snake phase2 tests passed");
    return 0;
}
