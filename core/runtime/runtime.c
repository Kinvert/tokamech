#include "core/runtime/runtime.h"

void tkm_runtime_init(
    TkmRuntime* runtime,
    void* env,
    TkmRuntimeObserveI16 observe,
    TkmRuntimeStepCommand step,
    const TkmIntBins* tokenizer,
    const TkmLookupPolicy* policy,
    const TkmDiscreteActionDecoder* decoder
) {
    runtime->env = env;
    runtime->observe = observe;
    runtime->step = step;
    runtime->tokenizer = tokenizer;
    runtime->policy = policy;
    runtime->decoder = decoder;
}

int tkm_runtime_run(TkmRuntime* runtime, uint32_t max_steps, TkmTrajectory* rollout) {
    for (uint32_t i = 0; i < max_steps; i++) {
        int16_t obs = 0;
        uint16_t token = 0;
        uint8_t action = 0;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (runtime->observe(runtime->env, &obs) != TKM_OK) {
            return TKM_ERR;
        }

        token = tkm_int_bins_encode(runtime->tokenizer, obs);
        if (token == TKM_INVALID_TOKEN) {
            return TKM_ERR;
        }

        action = tkm_lookup_policy_predict(runtime->policy, token);
        if (tkm_discrete_action_decode(runtime->decoder, action, &command) != TKM_OK) {
            return TKM_ERR;
        }
        if (runtime->step(runtime->env, command, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_trajectory_append(rollout, obs, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (terminal) {
            break;
        }
    }

    return TKM_OK;
}
