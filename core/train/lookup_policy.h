#ifndef TKM_LOOKUP_POLICY_H
#define TKM_LOOKUP_POLICY_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"

#define TKM_LOOKUP_MAX_TOKENS 256u
#define TKM_LOOKUP_MAX_ACTIONS 16u

typedef struct {
    uint8_t action_for_token[TKM_LOOKUP_MAX_TOKENS];
    uint8_t has_action[TKM_LOOKUP_MAX_TOKENS];
    uint16_t vocab_size;
    uint8_t num_actions;
} TkmLookupPolicy;

int tkm_lookup_policy_train(
    TkmLookupPolicy* policy,
    const TkmIntBins* tokenizer,
    const TkmTrajectory* trajectory,
    uint8_t num_actions
);

uint8_t tkm_lookup_policy_predict(const TkmLookupPolicy* policy, uint16_t token);

#endif
