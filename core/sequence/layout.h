#ifndef TKM_SEQUENCE_LAYOUT_H
#define TKM_SEQUENCE_LAYOUT_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_SEQUENCE_PAD 0
#define TKM_SEQUENCE_BOS 1
#define TKM_SEQUENCE_EOS 2
#define TKM_SEQUENCE_OBS 3
#define TKM_SEQUENCE_ACTION 4
#define TKM_SEQUENCE_REWARD 5
#define TKM_SEQUENCE_TERMINAL 6
#define TKM_SEQUENCE_TRANSITION 7

typedef struct {
    uint8_t type;
    int16_t token;
} TkmSequenceItem;

uint32_t tkm_sequence_required_obs_action_interleaved(uint32_t step_count);
uint32_t tkm_sequence_required_joint_transitions(uint32_t step_count);
int tkm_sequence_build_obs_action_interleaved(
    const int16_t* obs_tokens,
    const int16_t* action_tokens,
    const int16_t* reward_tokens,
    const int16_t* terminal_tokens,
    uint32_t step_count,
    TkmSequenceItem* out,
    uint32_t out_capacity,
    uint32_t* out_count
);
int tkm_sequence_build_joint_transitions(
    const int16_t* transition_tokens,
    uint32_t step_count,
    TkmSequenceItem* out,
    uint32_t out_capacity,
    uint32_t* out_count
);

#endif
