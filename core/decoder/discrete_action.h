#ifndef TKM_DISCRETE_ACTION_H
#define TKM_DISCRETE_ACTION_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/train/lookup_policy.h"

typedef struct {
    int8_t command_for_action[TKM_LOOKUP_MAX_ACTIONS];
    uint8_t has_action[TKM_LOOKUP_MAX_ACTIONS];
} TkmDiscreteActionDecoder;

int tkm_discrete_action_decoder_init(TkmDiscreteActionDecoder* decoder);
int tkm_discrete_action_decoder_set(TkmDiscreteActionDecoder* decoder, uint8_t action, int8_t command);
int tkm_discrete_action_decode(const TkmDiscreteActionDecoder* decoder, uint8_t action, int8_t* command);

#endif
