#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/centerline/centerline.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_int_bins_tokenizer_encodes_offsets(void) {
    TkmIntBins tokenizer;

    CHECK(tkm_int_bins_init(&tokenizer, -4, 4) == TKM_OK);
    CHECK(tokenizer.vocab_size == 9);
    CHECK(tkm_int_bins_encode(&tokenizer, -4) == 0);
    CHECK(tkm_int_bins_encode(&tokenizer, -3) == 1);
    CHECK(tkm_int_bins_encode(&tokenizer, 0) == 4);
    CHECK(tkm_int_bins_encode(&tokenizer, 4) == 8);
    CHECK(tkm_int_bins_encode(&tokenizer, 5) == TKM_INVALID_TOKEN);
}

static void test_centerline_oracle_collection_writes_replayable_trajectory(void) {
    CenterlineConfig cfg = {
        .start_offset = -3,
        .limit = 4,
        .horizon = 6,
    };
    TkmTrajectory traj;

    tkm_trajectory_init(&traj);
    CHECK(centerline_collect_oracle(&cfg, &traj) == TKM_OK);

    CHECK(traj.len == 6);
    CHECK(traj.obs_i16[0] == -3);
    CHECK(traj.actions[0] == CENTERLINE_ACTION_RIGHT);
    CHECK(traj.obs_i16[1] == -2);
    CHECK(traj.actions[1] == CENTERLINE_ACTION_RIGHT);
    CHECK(traj.obs_i16[2] == -1);
    CHECK(traj.actions[2] == CENTERLINE_ACTION_RIGHT);
    CHECK(traj.obs_i16[3] == 0);
    CHECK(traj.actions[3] == CENTERLINE_ACTION_STAY);
    CHECK(traj.terminals[5] == 1);

    TkmTrajectoryCursor cursor;
    int16_t obs = 99;
    uint8_t action = 99;
    float reward = 0.0f;
    uint8_t terminal = 0;

    tkm_trajectory_cursor_init(&cursor, &traj);
    CHECK(tkm_trajectory_cursor_next(&cursor, &obs, &action, &reward, &terminal) == TKM_OK);
    CHECK(obs == -3);
    CHECK(action == CENTERLINE_ACTION_RIGHT);
    CHECK(terminal == 0);
}

static void add_oracle_episode(TkmTrajectory* traj, int16_t start_offset) {
    CenterlineConfig cfg = {
        .start_offset = start_offset,
        .limit = 4,
        .horizon = 6,
    };

    CHECK(centerline_collect_oracle(&cfg, traj) == TKM_OK);
}

static void test_lookup_policy_learns_oracle_actions_from_tokens(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;

    tkm_trajectory_init(&dataset);
    CHECK(tkm_int_bins_init(&tokenizer, -4, 4) == TKM_OK);

    for (int16_t start = -3; start <= 3; start++) {
        add_oracle_episode(&dataset, start);
    }

    CHECK(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, CENTERLINE_ACTION_COUNT) == TKM_OK);

    CHECK(tkm_lookup_policy_predict(&policy, tkm_int_bins_encode(&tokenizer, -2)) == CENTERLINE_ACTION_RIGHT);
    CHECK(tkm_lookup_policy_predict(&policy, tkm_int_bins_encode(&tokenizer, 0)) == CENTERLINE_ACTION_STAY);
    CHECK(tkm_lookup_policy_predict(&policy, tkm_int_bins_encode(&tokenizer, 2)) == CENTERLINE_ACTION_LEFT);
}

static void test_runtime_closes_loop_with_trained_policy_and_decoder(void) {
    TkmTrajectory dataset;
    TkmTrajectory rollout;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    TkmDiscreteActionDecoder decoder;
    CenterlineEnv env;
    TkmRuntime runtime;

    tkm_trajectory_init(&dataset);
    tkm_trajectory_init(&rollout);
    CHECK(tkm_int_bins_init(&tokenizer, -4, 4) == TKM_OK);

    for (int16_t start = -3; start <= 3; start++) {
        add_oracle_episode(&dataset, start);
    }

    CHECK(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, CENTERLINE_ACTION_COUNT) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_init(&decoder) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_LEFT, -1) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_STAY, 0) == TKM_OK);
    CHECK(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_RIGHT, 1) == TKM_OK);

    centerline_env_init(&env, -3, 4, 8);
    tkm_runtime_init(
        &runtime,
        &env,
        centerline_observe_i16,
        centerline_step_command,
        &tokenizer,
        &policy,
        &decoder
    );

    CHECK(tkm_runtime_run(&runtime, 8, &rollout) == TKM_OK);
    CHECK(env.offset == 0);
    CHECK(rollout.len == 8);
    CHECK(rollout.obs_i16[0] == -3);
    CHECK(rollout.actions[0] == CENTERLINE_ACTION_RIGHT);
    CHECK(rollout.obs_i16[3] == 0);
    CHECK(rollout.actions[3] == CENTERLINE_ACTION_STAY);
    CHECK(rollout.terminals[7] == 1);
}

int main(void) {
    test_int_bins_tokenizer_encodes_offsets();
    test_centerline_oracle_collection_writes_replayable_trajectory();
    test_lookup_policy_learns_oracle_actions_from_tokens();
    test_runtime_closes_loop_with_trained_policy_and_decoder();
    puts("centerline phase0 tests passed");
    return 0;
}
