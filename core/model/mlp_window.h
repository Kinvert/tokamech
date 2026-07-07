#ifndef TKM_MLP_WINDOW_H
#define TKM_MLP_WINDOW_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_MLP_WINDOW_MAX_INPUT 256
#define TKM_MLP_WINDOW_MAX_HIDDEN 256
#define TKM_MLP_WINDOW_MAX_OUTPUT 256

typedef struct {
    uint32_t input_dim;
    uint32_t hidden_dim;
    uint32_t output_dim;
    float w0[TKM_MLP_WINDOW_MAX_INPUT * TKM_MLP_WINDOW_MAX_HIDDEN];
    float b0[TKM_MLP_WINDOW_MAX_HIDDEN];
    float w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float b1[TKM_MLP_WINDOW_MAX_OUTPUT];
} TkmMlpWindow;

int tkm_mlp_window_init(
    TkmMlpWindow* mlp,
    uint32_t input_dim,
    uint32_t hidden_dim,
    uint32_t output_dim,
    const float* w0,
    const float* b0,
    const float* w1,
    const float* b1
);
int tkm_mlp_window_forward(const TkmMlpWindow* mlp, const float* input, float* output, uint32_t output_count);
int tkm_mlp_window_train_mse(
    TkmMlpWindow* mlp,
    const float* input,
    uint16_t target_index,
    float learning_rate
);
int tkm_mlp_window_train_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    uint16_t target_index,
    float learning_rate
);
int tkm_mlp_window_train_masked_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    const uint8_t* valid_mask,
    uint16_t target_index,
    float learning_rate
);

#endif
