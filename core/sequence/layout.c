#include "core/sequence/layout.h"

uint32_t tkm_sequence_required_obs_action_interleaved(uint32_t step_count) {
    return 2u + step_count * 4u;
}

uint32_t tkm_sequence_required_joint_transitions(uint32_t step_count) {
    return 2u + step_count;
}

static void tkm_sequence_write(TkmSequenceItem* out, uint32_t index, uint8_t type, int16_t token) {
    out[index].type = type;
    out[index].token = token;
}

int tkm_sequence_build_obs_action_interleaved(
    const int16_t* obs_tokens,
    const int16_t* action_tokens,
    const int16_t* reward_tokens,
    const int16_t* terminal_tokens,
    uint32_t step_count,
    TkmSequenceItem* out,
    uint32_t out_capacity,
    uint32_t* out_count
) {
    uint32_t index = 0;
    uint32_t required = tkm_sequence_required_obs_action_interleaved(step_count);

    if (!obs_tokens || !action_tokens || !reward_tokens || !terminal_tokens || !out || !out_count || step_count == 0) {
        return TKM_ERR;
    }
    if (out_capacity < required) {
        return TKM_ERR;
    }

    tkm_sequence_write(out, index++, TKM_SEQUENCE_BOS, 0);
    for (uint32_t i = 0; i < step_count; i++) {
        tkm_sequence_write(out, index++, TKM_SEQUENCE_OBS, obs_tokens[i]);
        tkm_sequence_write(out, index++, TKM_SEQUENCE_ACTION, action_tokens[i]);
        tkm_sequence_write(out, index++, TKM_SEQUENCE_REWARD, reward_tokens[i]);
        tkm_sequence_write(out, index++, TKM_SEQUENCE_TERMINAL, terminal_tokens[i]);
    }
    tkm_sequence_write(out, index++, TKM_SEQUENCE_EOS, 0);

    *out_count = index;
    return TKM_OK;
}

int tkm_sequence_build_joint_transitions(
    const int16_t* transition_tokens,
    uint32_t step_count,
    TkmSequenceItem* out,
    uint32_t out_capacity,
    uint32_t* out_count
) {
    uint32_t index = 0;
    uint32_t required = tkm_sequence_required_joint_transitions(step_count);

    if (!transition_tokens || !out || !out_count || step_count == 0) {
        return TKM_ERR;
    }
    if (out_capacity < required) {
        return TKM_ERR;
    }

    tkm_sequence_write(out, index++, TKM_SEQUENCE_BOS, 0);
    for (uint32_t i = 0; i < step_count; i++) {
        tkm_sequence_write(out, index++, TKM_SEQUENCE_TRANSITION, transition_tokens[i]);
    }
    tkm_sequence_write(out, index++, TKM_SEQUENCE_EOS, 0);

    *out_count = index;
    return TKM_OK;
}
