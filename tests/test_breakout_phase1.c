#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/breakout/breakout.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_breakout_reset_matches_pufferlib_shape(void) {
    BreakoutEnv env;

    breakout_env_init_default(&env);
    breakout_reset(&env);

    CHECK(env.width == 576);
    CHECK(env.height == 330);
    CHECK(env.brick_rows == 6);
    CHECK(env.brick_cols == 18);
    CHECK(env.num_bricks == 108);
    CHECK(BREAKOUT_OBS_SIZE == 118);
    CHECK(env.actions[0] == BREAKOUT_NOOP);
    CHECK(env.actions[1] == BREAKOUT_LEFT);
    CHECK(env.actions[2] == BREAKOUT_RIGHT);
    CHECK(env.num_balls == 5);
    CHECK(env.score == 0);
    CHECK(env.observations[0] > 0.4f && env.observations[0] < 0.6f);
    CHECK(env.observations[2] > 0.4f && env.observations[2] < 0.6f);
    CHECK(env.observations[10] == 0.0f);
    CHECK(env.observations[117] == 0.0f);
}

static void test_breakout_paddle_actions_and_tokenizer(void) {
    BreakoutEnv env;
    float start_x;
    uint16_t token_left;
    uint16_t token_center;
    uint16_t token_right;

    breakout_env_init_default(&env);
    breakout_reset(&env);

    start_x = env.paddle_x;
    CHECK(breakout_step(&env, BREAKOUT_LEFT) == TKM_OK);
    CHECK(env.paddle_x < start_x);

    env.ball_x = env.paddle_x - 80.0f;
    breakout_compute_observations(&env);
    token_left = breakout_tokenize_observation(env.observations);

    env.ball_x = env.paddle_x + env.paddle_width * 0.5f - env.ball_width * 0.5f;
    breakout_compute_observations(&env);
    token_center = breakout_tokenize_observation(env.observations);

    env.ball_x = env.paddle_x + env.paddle_width + 80.0f;
    breakout_compute_observations(&env);
    token_right = breakout_tokenize_observation(env.observations);

    CHECK(token_left != token_center);
    CHECK(token_center != token_right);
    CHECK(breakout_exploration_action(0u) < BREAKOUT_ACTION_COUNT);
    CHECK(breakout_exploration_action(11u) < BREAKOUT_ACTION_COUNT);
}

static void test_breakout_brick_collision_rewards_and_terminal(void) {
    BreakoutEnv env;
    uint8_t terminal = 0;
    float reward = 0.0f;

    breakout_env_init_default(&env);
    breakout_reset(&env);

    env.balls_fired = 1;
    env.ball_x = env.brick_x[0] + 4.0f;
    env.ball_y = env.brick_y[0] + env.brick_height + 1.0f;
    env.ball_vx = 0.0f;
    env.ball_vy = -4.0f;

    CHECK(breakout_step_env(&env, BREAKOUT_NOOP, &reward, &terminal) == TKM_OK);
    CHECK(reward > 0.0f);
    CHECK(env.score > 0);
    CHECK(env.brick_states[0] == 1.0f);
    CHECK(terminal == 0);

    env.ball_y = env.height + 10.0f;
    env.num_balls = 0;
    CHECK(breakout_step_env(&env, BREAKOUT_NOOP, &reward, &terminal) == TKM_OK);
    CHECK(terminal == 1);
}

static void test_breakout_token_exploration_collection_and_lookup_policy(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    TkmDiscreteActionDecoder decoder;
    BreakoutEnv env;
    uint16_t left_token;
    uint16_t center_token;
    uint16_t right_token;
    uint8_t action;
    int8_t command;

    tkm_trajectory_init(&dataset);
    CHECK(breakout_collect_exploration_tokens(&dataset) == TKM_OK);
    CHECK(dataset.len > 12);
    CHECK(breakout_policy_tokenizer_init(&tokenizer) == TKM_OK);
    CHECK(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, BREAKOUT_ACTION_COUNT) == TKM_OK);

    CHECK(tkm_discrete_action_decoder_init(&decoder) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, BREAKOUT_NOOP, BREAKOUT_NOOP) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, BREAKOUT_LEFT, BREAKOUT_LEFT) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, BREAKOUT_RIGHT, BREAKOUT_RIGHT) == TKM_OK);

    left_token = breakout_make_policy_token(-2);
    center_token = breakout_make_policy_token(0);
    right_token = breakout_make_policy_token(2);

    CHECK(tkm_lookup_policy_predict(&policy, left_token) < BREAKOUT_ACTION_COUNT);
    CHECK(tkm_lookup_policy_predict(&policy, center_token) < BREAKOUT_ACTION_COUNT);
    CHECK(tkm_lookup_policy_predict(&policy, right_token) < BREAKOUT_ACTION_COUNT);

    breakout_env_init_default(&env);
    breakout_reset(&env);
    env.ball_x = env.paddle_x + env.paddle_width + 90.0f;
    breakout_compute_observations(&env);

    action = tkm_lookup_policy_predict(&policy, breakout_tokenize_observation(env.observations));
    CHECK(tkm_discrete_action_decode(&decoder, action, &command) == TKM_OK);
    CHECK(command >= 0);
}

static void test_breakout_runtime_hits_paddle_on_fixed_rollout(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    BreakoutEnv env;
    BreakoutRolloutStats stats;

    tkm_trajectory_init(&dataset);
    CHECK(breakout_collect_exploration_tokens(&dataset) == TKM_OK);
    CHECK(breakout_policy_tokenizer_init(&tokenizer) == TKM_OK);
    CHECK(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, BREAKOUT_ACTION_COUNT) == TKM_OK);

    breakout_env_init_default(&env);
    breakout_reset(&env);
    CHECK(breakout_run_token_policy(&env, &policy, 220, &stats) == TKM_OK);
    CHECK(stats.steps > 50);
    CHECK(stats.paddle_hits <= stats.steps);
    CHECK(stats.terminal == 0);
}

int main(void) {
    test_breakout_reset_matches_pufferlib_shape();
    test_breakout_paddle_actions_and_tokenizer();
    test_breakout_brick_collision_rewards_and_terminal();
    test_breakout_token_exploration_collection_and_lookup_policy();
    test_breakout_runtime_hits_paddle_on_fixed_rollout();
    puts("breakout phase1 tests passed");
    return 0;
}
