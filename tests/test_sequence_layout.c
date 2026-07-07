#include <stdio.h>
#include <stdlib.h>

#include "core/sequence/layout.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void check_item(TkmSequenceItem item, uint8_t type, int16_t token) {
    CHECK(item.type == type);
    CHECK(item.token == token);
}

static void test_obs_action_interleaved_layout_writes_typed_items(void) {
    const int16_t obs[2] = {10, 11};
    const int16_t actions[2] = {1, 2};
    const int16_t rewards[2] = {5, -5};
    const int16_t terminals[2] = {0, 1};
    TkmSequenceItem out[10];
    uint32_t out_count = 0;

    CHECK(tkm_sequence_required_obs_action_interleaved(2) == 10);
    CHECK(tkm_sequence_build_obs_action_interleaved(obs, actions, rewards, terminals, 2, out, 10, &out_count) == TKM_OK);
    CHECK(out_count == 10);
    check_item(out[0], TKM_SEQUENCE_BOS, 0);
    check_item(out[1], TKM_SEQUENCE_OBS, 10);
    check_item(out[2], TKM_SEQUENCE_ACTION, 1);
    check_item(out[3], TKM_SEQUENCE_REWARD, 5);
    check_item(out[4], TKM_SEQUENCE_TERMINAL, 0);
    check_item(out[5], TKM_SEQUENCE_OBS, 11);
    check_item(out[6], TKM_SEQUENCE_ACTION, 2);
    check_item(out[7], TKM_SEQUENCE_REWARD, -5);
    check_item(out[8], TKM_SEQUENCE_TERMINAL, 1);
    check_item(out[9], TKM_SEQUENCE_EOS, 0);
}

static void test_joint_transition_layout_keeps_one_token_per_step(void) {
    const int16_t transitions[3] = {101, 102, 103};
    TkmSequenceItem out[5];
    uint32_t out_count = 0;

    CHECK(tkm_sequence_required_joint_transitions(3) == 5);
    CHECK(tkm_sequence_build_joint_transitions(transitions, 3, out, 5, &out_count) == TKM_OK);
    CHECK(out_count == 5);
    check_item(out[0], TKM_SEQUENCE_BOS, 0);
    check_item(out[1], TKM_SEQUENCE_TRANSITION, 101);
    check_item(out[2], TKM_SEQUENCE_TRANSITION, 102);
    check_item(out[3], TKM_SEQUENCE_TRANSITION, 103);
    check_item(out[4], TKM_SEQUENCE_EOS, 0);
}

static void test_sequence_layout_rejects_bad_inputs_and_short_buffers(void) {
    const int16_t values[1] = {1};
    TkmSequenceItem out[4];
    uint32_t out_count = 99;

    CHECK(tkm_sequence_build_obs_action_interleaved(0, values, values, values, 1, out, 6, &out_count) == TKM_ERR);
    CHECK(tkm_sequence_build_obs_action_interleaved(values, values, values, values, 0, out, 6, &out_count) == TKM_ERR);
    CHECK(tkm_sequence_build_obs_action_interleaved(values, values, values, values, 1, out, 5, &out_count) == TKM_ERR);
    CHECK(tkm_sequence_build_obs_action_interleaved(values, values, values, values, 1, out, 6, 0) == TKM_ERR);
    CHECK(tkm_sequence_build_joint_transitions(0, 1, out, 3, &out_count) == TKM_ERR);
    CHECK(tkm_sequence_build_joint_transitions(values, 0, out, 3, &out_count) == TKM_ERR);
    CHECK(tkm_sequence_build_joint_transitions(values, 1, out, 2, &out_count) == TKM_ERR);
}

int main(void) {
    test_obs_action_interleaved_layout_writes_typed_items();
    test_joint_transition_layout_keeps_one_token_per_step();
    test_sequence_layout_rejects_bad_inputs_and_short_buffers();
    puts("sequence layout tests passed");
    return 0;
}
