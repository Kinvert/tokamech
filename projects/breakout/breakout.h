#ifndef TKM_BREAKOUT_H
#define TKM_BREAKOUT_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"

#define BREAKOUT_NOOP 0
#define BREAKOUT_LEFT 1
#define BREAKOUT_RIGHT 2
#define BREAKOUT_ACTION_COUNT 3

#define BREAKOUT_WIDTH 576
#define BREAKOUT_HEIGHT 330
#define BREAKOUT_BRICK_ROWS 6
#define BREAKOUT_BRICK_COLS 18
#define BREAKOUT_NUM_BRICKS (BREAKOUT_BRICK_ROWS * BREAKOUT_BRICK_COLS)
#define BREAKOUT_OBS_SIZE (10 + BREAKOUT_NUM_BRICKS)

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} BreakoutLog;

typedef struct {
    BreakoutLog log;
    float observations[BREAKOUT_OBS_SIZE];
    uint8_t actions[BREAKOUT_ACTION_COUNT];
    float rewards[1];
    uint8_t terminals[1];
    int score;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float brick_x[BREAKOUT_NUM_BRICKS];
    float brick_y[BREAKOUT_NUM_BRICKS];
    float brick_states[BREAKOUT_NUM_BRICKS];
    int balls_fired;
    float initial_paddle_width;
    float paddle_width;
    float paddle_height;
    float paddle_speed;
    float ball_speed;
    float initial_ball_speed;
    float max_ball_speed;
    int hits;
    int paddle_hits;
    int width;
    int height;
    int num_bricks;
    int brick_rows;
    int brick_cols;
    int ball_width;
    int ball_height;
    int brick_width;
    int brick_height;
    int num_balls;
    int max_score;
    int half_max_score;
    int tick;
    int frameskip;
    uint8_t hit_brick;
    uint8_t last_hit_paddle;
    uint8_t terminal;
} BreakoutEnv;

typedef struct {
    uint32_t steps;
    uint32_t paddle_hits;
    int score;
    uint8_t terminal;
} BreakoutRolloutStats;

void breakout_env_init_default(BreakoutEnv* env);
void breakout_reset(BreakoutEnv* env);
void breakout_compute_observations(BreakoutEnv* env);
int breakout_step(BreakoutEnv* env, uint8_t action);
int breakout_step_env(BreakoutEnv* env, uint8_t action, float* reward, uint8_t* terminal);

uint16_t breakout_make_policy_token(int relative_bucket);
uint16_t breakout_tokenize_observation(const float* observations);
uint8_t breakout_oracle_action_from_token(uint16_t token);
uint8_t breakout_oracle_action(const BreakoutEnv* env);
int breakout_policy_tokenizer_init(TkmIntBins* tokenizer);
int breakout_collect_oracle_tokens(TkmTrajectory* trajectory);
int breakout_run_token_policy(
    BreakoutEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    BreakoutRolloutStats* stats
);

#endif
