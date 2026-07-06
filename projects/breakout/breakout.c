#include "projects/breakout/breakout.h"

#include <math.h>
#include <string.h>

#define BREAKOUT_Y_OFFSET 50.0f
#define BREAKOUT_TICK_RATE (1.0f / 60.0f)
#define BREAKOUT_HALF_PADDLE_WIDTH 31.0f

static float breakout_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static uint8_t breakout_rects_overlap(
    float ax,
    float ay,
    float aw,
    float ah,
    float bx,
    float by,
    float bw,
    float bh
) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

static void breakout_generate_bricks(BreakoutEnv* env) {
    env->half_max_score = 0;
    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            env->brick_x[idx] = (float)(col * env->brick_width);
            env->brick_y[idx] = (float)(row * env->brick_height) + BREAKOUT_Y_OFFSET;
            env->brick_states[idx] = 0.0f;
            env->half_max_score += 7 - 3 * (row / 2);
        }
    }
    env->max_score = 2 * env->half_max_score;
}

void breakout_env_init_default(BreakoutEnv* env) {
    memset(env, 0, sizeof(*env));
    env->frameskip = 1;
    env->width = BREAKOUT_WIDTH;
    env->height = BREAKOUT_HEIGHT;
    env->initial_paddle_width = 62.0f;
    env->paddle_width = 62.0f;
    env->paddle_height = 8.0f;
    env->ball_width = 32;
    env->ball_height = 32;
    env->brick_width = 32;
    env->brick_height = 12;
    env->brick_rows = BREAKOUT_BRICK_ROWS;
    env->brick_cols = BREAKOUT_BRICK_COLS;
    env->num_bricks = BREAKOUT_NUM_BRICKS;
    env->initial_ball_speed = 256.0f;
    env->max_ball_speed = 448.0f;
    env->paddle_speed = 620.0f;
    env->actions[0] = BREAKOUT_NOOP;
    env->actions[1] = BREAKOUT_LEFT;
    env->actions[2] = BREAKOUT_RIGHT;
    breakout_generate_bricks(env);
}

static void breakout_reset_round(BreakoutEnv* env) {
    env->balls_fired = 0;
    env->hit_brick = 0;
    env->last_hit_paddle = 0;
    env->hits = 0;
    env->ball_speed = env->initial_ball_speed;
    env->paddle_width = env->initial_paddle_width;
    env->paddle_x = (float)env->width / 2.0f - env->paddle_width / 2.0f;
    env->paddle_y = (float)env->height - env->paddle_height - 10.0f;
    env->ball_x = env->paddle_x + (env->paddle_width / 2.0f - (float)env->ball_width / 2.0f);
    env->ball_y = (float)env->height / 2.0f - 30.0f;
    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
}

void breakout_reset(BreakoutEnv* env) {
    env->score = 0;
    env->num_balls = 5;
    env->tick = 0;
    env->terminal = 0;
    env->paddle_hits = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    for (int i = 0; i < env->num_bricks; i++) {
        env->brick_states[i] = 0.0f;
    }
    breakout_reset_round(env);
    breakout_compute_observations(env);
}

void breakout_compute_observations(BreakoutEnv* env) {
    env->observations[0] = env->paddle_x / (float)env->width;
    env->observations[1] = env->paddle_y / (float)env->height;
    env->observations[2] = env->ball_x / (float)env->width;
    env->observations[3] = env->ball_y / (float)env->height;
    env->observations[4] = env->ball_vx / 512.0f;
    env->observations[5] = env->ball_vy / 512.0f;
    env->observations[6] = (float)env->balls_fired / 5.0f;
    env->observations[7] = (float)env->score / 864.0f;
    env->observations[8] = (float)env->num_balls / 5.0f;
    env->observations[9] = env->paddle_width / (2.0f * BREAKOUT_HALF_PADDLE_WIDTH);
    memcpy(env->observations + 10, env->brick_states, sizeof(float) * BREAKOUT_NUM_BRICKS);
}

static int breakout_brick_points(const BreakoutEnv* env, int brick_idx) {
    int row = brick_idx / env->brick_cols;
    return 7 - 3 * (row / 2);
}

static void breakout_destroy_brick(BreakoutEnv* env, int brick_idx) {
    int points = breakout_brick_points(env, brick_idx);
    env->score += points;
    env->brick_states[brick_idx] = 1.0f;
    env->rewards[0] += (float)points;
    env->hit_brick = 1;
    if (brick_idx / env->brick_cols < 3) {
        env->ball_speed = env->max_ball_speed;
    }
}

static void breakout_launch_ball(BreakoutEnv* env) {
    env->balls_fired = 1;
    env->ball_vx = 2.8f;
    env->ball_vy = 4.0f;
}

static void breakout_move_paddle(BreakoutEnv* env, uint8_t action) {
    float act = 0.0f;
    if (action == BREAKOUT_LEFT) {
        act = -1.0f;
    } else if (action == BREAKOUT_RIGHT) {
        act = 1.0f;
    }

    env->paddle_x += act * env->paddle_speed * BREAKOUT_TICK_RATE;
    env->paddle_x = breakout_clampf(env->paddle_x, 0.0f, (float)env->width - env->paddle_width);
}

static void breakout_handle_walls(BreakoutEnv* env) {
    if (env->ball_x <= 0.0f) {
        env->ball_x = 0.0f;
        env->ball_vx = fabsf(env->ball_vx);
    }
    if (env->ball_x + (float)env->ball_width >= (float)env->width) {
        env->ball_x = (float)env->width - (float)env->ball_width;
        env->ball_vx = -fabsf(env->ball_vx);
    }
    if (env->ball_y <= 0.0f) {
        env->ball_y = 0.0f;
        env->ball_vy = fabsf(env->ball_vy);
    }
}

static void breakout_handle_paddle(BreakoutEnv* env) {
    if (env->ball_vy <= 0.0f) {
        return;
    }

    if (!breakout_rects_overlap(
            env->ball_x,
            env->ball_y,
            (float)env->ball_width,
            (float)env->ball_height,
            env->paddle_x,
            env->paddle_y,
            env->paddle_width,
            env->paddle_height)) {
        return;
    }

    float ball_center = env->ball_x + (float)env->ball_width * 0.5f;
    float relative = (ball_center - env->paddle_x) / env->paddle_width;
    float angle = -0.78539816339f + relative * 1.57079632679f;
    env->ball_vx = sinf(angle) * env->ball_speed * BREAKOUT_TICK_RATE;
    env->ball_vy = -fabsf(cosf(angle) * env->ball_speed * BREAKOUT_TICK_RATE);
    env->ball_y = env->paddle_y - (float)env->ball_height - 1.0f;
    env->hits++;
    env->paddle_hits++;
    env->last_hit_paddle = 1;
    if (env->hits % 4 == 0 && env->ball_speed < env->max_ball_speed) {
        env->ball_speed += 64.0f;
    }
}

static void breakout_handle_bricks(BreakoutEnv* env) {
    for (int i = 0; i < env->num_bricks; i++) {
        if (env->brick_states[i] != 0.0f) {
            continue;
        }

        if (breakout_rects_overlap(
                env->ball_x,
                env->ball_y,
                (float)env->ball_width,
                (float)env->ball_height,
                env->brick_x[i],
                env->brick_y[i],
                (float)env->brick_width,
                (float)env->brick_height)) {
            breakout_destroy_brick(env, i);
            env->ball_vy = env->ball_vy > 0.0f ? -fabsf(env->ball_vy) : fabsf(env->ball_vy);
            return;
        }
    }
}

int breakout_step_env(BreakoutEnv* env, uint8_t action, float* reward, uint8_t* terminal) {
    if (action >= BREAKOUT_ACTION_COUNT) {
        return TKM_ERR;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->last_hit_paddle = 0;

    if (env->terminal) {
        *reward = 0.0f;
        *terminal = 1;
        return TKM_OK;
    }

    for (int frame = 0; frame < env->frameskip; frame++) {
        env->tick++;
        if (env->balls_fired == 0) {
            breakout_launch_ball(env);
        }

        breakout_move_paddle(env, action);
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
        breakout_handle_walls(env);
        breakout_handle_paddle(env);
        breakout_handle_bricks(env);

        if (env->ball_y >= env->paddle_y + env->paddle_height) {
            env->num_balls--;
            if (env->num_balls < 0) {
                env->terminal = 1;
                env->terminals[0] = 1;
                break;
            }
            breakout_reset_round(env);
        }

        if (env->score >= env->max_score) {
            env->terminal = 1;
            env->terminals[0] = 1;
            break;
        }
    }

    breakout_compute_observations(env);
    *reward = env->rewards[0];
    *terminal = env->terminals[0];
    return TKM_OK;
}

int breakout_step(BreakoutEnv* env, uint8_t action) {
    float reward = 0.0f;
    uint8_t terminal = 0;
    return breakout_step_env(env, action, &reward, &terminal);
}

uint16_t breakout_make_policy_token(int relative_bucket) {
    if (relative_bucket < -2) {
        relative_bucket = -2;
    }
    if (relative_bucket > 2) {
        relative_bucket = 2;
    }
    return (uint16_t)(relative_bucket + 2);
}

uint16_t breakout_tokenize_observation(const float* observations) {
    float paddle_center = observations[0] + (31.0f / (float)BREAKOUT_WIDTH);
    float ball_center = observations[2] + (16.0f / (float)BREAKOUT_WIDTH);
    float diff = ball_center - paddle_center;

    if (diff < -0.14f) {
        return breakout_make_policy_token(-2);
    }
    if (diff < -0.035f) {
        return breakout_make_policy_token(-1);
    }
    if (diff > 0.14f) {
        return breakout_make_policy_token(2);
    }
    if (diff > 0.035f) {
        return breakout_make_policy_token(1);
    }
    return breakout_make_policy_token(0);
}

uint8_t breakout_oracle_action_from_token(uint16_t token) {
    if (token < breakout_make_policy_token(0)) {
        return BREAKOUT_LEFT;
    }
    if (token > breakout_make_policy_token(0)) {
        return BREAKOUT_RIGHT;
    }
    return BREAKOUT_NOOP;
}

uint8_t breakout_oracle_action(const BreakoutEnv* env) {
    return breakout_oracle_action_from_token(breakout_tokenize_observation(env->observations));
}

int breakout_policy_tokenizer_init(TkmIntBins* tokenizer) {
    return tkm_int_bins_init(tokenizer, 0, 4);
}

int breakout_collect_oracle_tokens(TkmTrajectory* trajectory) {
    BreakoutEnv env;
    breakout_env_init_default(&env);

    for (int bucket = -2; bucket <= 2; bucket++) {
        uint16_t token = breakout_make_policy_token(bucket);
        uint8_t action = breakout_oracle_action_from_token(token);
        for (int repeat = 0; repeat < 8; repeat++) {
            if (tkm_trajectory_append(trajectory, (int16_t)token, action, 0.0f, 0) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    breakout_reset(&env);
    for (uint32_t i = 0; i < 180 && !env.terminal; i++) {
        uint16_t token = breakout_tokenize_observation(env.observations);
        uint8_t action = breakout_oracle_action_from_token(token);
        float reward = 0.0f;
        uint8_t terminal = 0;
        if (breakout_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_trajectory_append(trajectory, (int16_t)token, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int breakout_run_token_policy(
    BreakoutEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    BreakoutRolloutStats* stats
) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint16_t token = breakout_tokenize_observation(env->observations);
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
