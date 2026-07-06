#ifndef TKM_RUNTIME_H
#define TKM_RUNTIME_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"

typedef int (*TkmRuntimeObserveI16)(void* env, int16_t* obs);
typedef int (*TkmRuntimeStepCommand)(void* env, int8_t command, float* reward, uint8_t* terminal);

typedef struct {
    void* env;
    TkmRuntimeObserveI16 observe;
    TkmRuntimeStepCommand step;
    const TkmIntBins* tokenizer;
    const TkmLookupPolicy* policy;
    const TkmDiscreteActionDecoder* decoder;
} TkmRuntime;

void tkm_runtime_init(
    TkmRuntime* runtime,
    void* env,
    TkmRuntimeObserveI16 observe,
    TkmRuntimeStepCommand step,
    const TkmIntBins* tokenizer,
    const TkmLookupPolicy* policy,
    const TkmDiscreteActionDecoder* decoder
);

int tkm_runtime_run(TkmRuntime* runtime, uint32_t max_steps, TkmTrajectory* rollout);

#endif
