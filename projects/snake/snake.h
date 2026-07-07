#ifndef TKM_SNAKE_H
#define TKM_SNAKE_H

#include <stdint.h>

#include "core/config/decoder_config.h"
#include "core/common/status.h"
#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "core/train/sparse_lookup_policy.h"

#define SNAKE_WIDTH 12
#define SNAKE_HEIGHT 12
#define SNAKE_MAX_CELLS (SNAKE_WIDTH * SNAKE_HEIGHT)
#define SNAKE_OBS_SIZE 11
#define SNAKE_FEATURE_COUNT (12 + SNAKE_MAX_CELLS)

#define SNAKE_EMPTY 0
#define SNAKE_FOOD 1
#define SNAKE_BODY 2
#define SNAKE_WALL 3

#define SNAKE_UP 0
#define SNAKE_DOWN 1
#define SNAKE_LEFT 2
#define SNAKE_RIGHT 3
#define SNAKE_ACTION_COUNT 4
#define SNAKE_ACTION_FEATURES_PER_ACTION 4
#define SNAKE_ACTION_FEATURE_COUNT (SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * SNAKE_ACTION_FEATURES_PER_ACTION)
#define SNAKE_ACTION_SPACE_FEATURES_PER_ACTION 5
#define SNAKE_ACTION_SPACE_FEATURE_COUNT (SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * SNAKE_ACTION_SPACE_FEATURES_PER_ACTION)
#define SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION 6
#define SNAKE_ACTION_FOOD_SPACE_FEATURE_COUNT (SNAKE_FEATURE_COUNT + SNAKE_ACTION_COUNT * SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION)
#define SNAKE_ACTION_MAX_FEATURES_PER_ACTION SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION
#define SNAKE_ACTION_MAX_FEATURE_COUNT SNAKE_ACTION_FOOD_SPACE_FEATURE_COUNT

#define SNAKE_TOKEN_COUNT 128
#define SNAKE_RICH_TOKEN_COUNT 40000u

typedef struct {
    int width;
    int height;
    uint8_t grid[SNAKE_MAX_CELLS];
    int16_t observations[SNAKE_OBS_SIZE];
    uint8_t actions[SNAKE_ACTION_COUNT];
    int16_t body_r[SNAKE_MAX_CELLS];
    int16_t body_c[SNAKE_MAX_CELLS];
    int length;
    int head_r;
    int head_c;
    int food_r;
    int food_c;
    uint8_t dir;
    int score;
    uint32_t tick;
    uint8_t terminal;
} SnakeEnv;

typedef struct {
    uint32_t steps;
    uint32_t food_eaten;
    int score;
    uint8_t terminal;
} SnakeRolloutStats;

void snake_env_init_default(SnakeEnv* env);
void snake_reset_fixed(SnakeEnv* env);
void snake_rebuild_grid(SnakeEnv* env);
void snake_compute_observations(SnakeEnv* env);
void snake_set_food(SnakeEnv* env, int row, int col);
int snake_step_env(SnakeEnv* env, uint8_t action, float* reward, uint8_t* terminal);
int snake_step(SnakeEnv* env, uint8_t action);
int snake_write_features(const SnakeEnv* env, float* out_features, uint32_t max_features);
int snake_write_action_features(const SnakeEnv* env, float* out_features, uint32_t max_features);
int snake_write_action_space_features(const SnakeEnv* env, float* out_features, uint32_t max_features);
int snake_write_action_food_space_features(const SnakeEnv* env, float* out_features, uint32_t max_features);
int snake_write_action_mask_immediate(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions);
int snake_write_action_mask_survival(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions);
int snake_write_action_mask(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions);
uint8_t snake_filter_unsafe_action(const SnakeEnv* env, uint8_t suggested_action, uint8_t fallback_action);
uint8_t snake_filter_planner_score_action(const SnakeEnv* env, uint8_t suggested_action);
uint8_t snake_filter_best_score_action(const SnakeEnv* env);
uint8_t snake_filter_rollout_score_action(const SnakeEnv* env, uint8_t suggested_action, uint32_t horizon);
uint8_t snake_filter_rollout_score_mode_action(
    const SnakeEnv* env,
    uint8_t suggested_action,
    uint32_t horizon,
    TkmDecoderRolloutScoreKind score_mode
);
uint8_t snake_decode_action(const SnakeEnv* env, const TkmDecoderConfig* decoder, uint8_t suggested_action);

uint16_t snake_tokenize_observation(const SnakeEnv* env);
uint16_t snake_tokenize_rich_observation(const SnakeEnv* env);
int32_t snake_tokenize_full_observation(const SnakeEnv* env);
int snake_policy_tokenizer_init(TkmIntBins* tokenizer);
int snake_rich_policy_tokenizer_init(TkmIntBins* tokenizer);
uint8_t snake_oracle_action(const SnakeEnv* env);
uint8_t snake_planner_action(const SnakeEnv* env);
int snake_collect_oracle_tokens(TkmTrajectory* trajectory);
int snake_collect_planner_tokens(TkmTrajectory* trajectory, uint32_t max_steps);
int snake_collect_planner_tokens_for_food(TkmTrajectory* trajectory, int food_r, int food_c, uint32_t max_steps);
int snake_run_token_policy(
    SnakeEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
);
int snake_run_rich_token_policy(
    SnakeEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
);
int snake_run_sparse_token_policy(
    SnakeEnv* env,
    const TkmSparseLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
);
int snake_run_decoder_config_policy(
    SnakeEnv* env,
    const TkmDecoderConfig* decoder,
    uint32_t max_steps,
    SnakeRolloutStats* stats
);
int snake_run_planner_policy(SnakeEnv* env, uint32_t max_steps, SnakeRolloutStats* stats);

#endif
