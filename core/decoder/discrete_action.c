#include "core/decoder/discrete_action.h"

int tkm_discrete_action_decoder_init(TkmDiscreteActionDecoder* decoder) {
    for (uint8_t i = 0; i < TKM_LOOKUP_MAX_ACTIONS; i++) {
        decoder->command_for_action[i] = 0;
        decoder->has_action[i] = 0;
    }

    return TKM_OK;
}

int tkm_discrete_action_decoder_set(TkmDiscreteActionDecoder* decoder, uint8_t action, int8_t command) {
    if (action >= TKM_LOOKUP_MAX_ACTIONS) {
        return TKM_ERR;
    }

    decoder->command_for_action[action] = command;
    decoder->has_action[action] = 1;
    return TKM_OK;
}

int tkm_discrete_action_decode(const TkmDiscreteActionDecoder* decoder, uint8_t action, int8_t* command) {
    if (action >= TKM_LOOKUP_MAX_ACTIONS || !decoder->has_action[action]) {
        return TKM_ERR;
    }

    *command = decoder->command_for_action[action];
    return TKM_OK;
}
