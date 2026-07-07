#ifndef TKM_LOSS_H
#define TKM_LOSS_H

#include <stdint.h>

#include "core/common/status.h"

int tkm_loss_cross_entropy(const float* probabilities, uint32_t count, uint32_t target, float* out_loss);
int tkm_loss_mse(const float* predicted, const float* target, uint32_t count, float* out_loss);
int tkm_loss_huber(const float* predicted, const float* target, uint32_t count, float delta, float* out_loss);
int tkm_loss_binary_cross_entropy(float probability, uint8_t target, float* out_loss);
int tkm_loss_weighted_sum(const float* losses, const float* weights, uint32_t count, float* out_loss);
int tkm_loss_action_smoothness(const float* actions, uint32_t step_count, uint32_t action_dim, float* out_loss);

#endif
