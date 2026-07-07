#include "core/model/mlp_window.h"

#include <math.h>

static uint32_t tkm_mlp_w0_offset(const TkmMlpWindow* mlp, uint32_t input_index, uint32_t hidden_index) {
    return input_index * mlp->hidden_dim + hidden_index;
}

static uint32_t tkm_mlp_w1_offset(const TkmMlpWindow* mlp, uint32_t hidden_index, uint32_t output_index) {
    return hidden_index * mlp->output_dim + output_index;
}

static float tkm_mlp_relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

static int tkm_mlp_ready(const TkmMlpWindow* mlp) {
    return mlp && mlp->input_dim > 0 && mlp->hidden_dim > 0 && mlp->output_dim > 0;
}

int tkm_mlp_window_init(
    TkmMlpWindow* mlp,
    uint32_t input_dim,
    uint32_t hidden_dim,
    uint32_t output_dim,
    const float* w0,
    const float* b0,
    const float* w1,
    const float* b1
) {
    if (!mlp || !w0 || !w1 || input_dim == 0 || hidden_dim == 0 || output_dim == 0) {
        return TKM_ERR;
    }
    if (input_dim > TKM_MLP_WINDOW_MAX_INPUT ||
        hidden_dim > TKM_MLP_WINDOW_MAX_HIDDEN ||
        output_dim > TKM_MLP_WINDOW_MAX_OUTPUT) {
        return TKM_ERR;
    }

    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;

    for (uint32_t i = 0; i < input_dim; i++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            mlp->w0[tkm_mlp_w0_offset(mlp, i, h)] = w0[i * hidden_dim + h];
        }
    }

    for (uint32_t h = 0; h < hidden_dim; h++) {
        mlp->b0[h] = b0 ? b0[h] : 0.0f;
        for (uint32_t o = 0; o < output_dim; o++) {
            mlp->w1[tkm_mlp_w1_offset(mlp, h, o)] = w1[h * output_dim + o];
        }
    }

    for (uint32_t o = 0; o < output_dim; o++) {
        mlp->b1[o] = b1 ? b1[o] : 0.0f;
    }

    return TKM_OK;
}

int tkm_mlp_window_forward(const TkmMlpWindow* mlp, const float* input, float* output, uint32_t output_count) {
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];

    if (!tkm_mlp_ready(mlp) || !input || !output || output_count < mlp->output_dim) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float value = mlp->b0[h];
        for (uint32_t i = 0; i < mlp->input_dim; i++) {
            value += input[i] * mlp->w0[tkm_mlp_w0_offset(mlp, i, h)];
        }
        hidden[h] = tkm_mlp_relu(value);
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float value = mlp->b1[o];
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            value += hidden[h] * mlp->w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
        output[o] = value;
    }

    return TKM_OK;
}

int tkm_mlp_window_train_mse(
    TkmMlpWindow* mlp,
    const float* input,
    uint16_t target_index,
    float learning_rate
) {
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float output[TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_output[TKM_MLP_WINDOW_MAX_OUTPUT];
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];

    if (!tkm_mlp_ready(mlp) || !input || target_index >= mlp->output_dim || learning_rate <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float value = mlp->b0[h];
        for (uint32_t i = 0; i < mlp->input_dim; i++) {
            value += input[i] * mlp->w0[tkm_mlp_w0_offset(mlp, i, h)];
        }
        pre_hidden[h] = value;
        hidden[h] = tkm_mlp_relu(value);
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            old_w1[tkm_mlp_w1_offset(mlp, h, o)] = mlp->w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float value = mlp->b1[o];
        float target = o == target_index ? 1.0f : 0.0f;
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            value += hidden[h] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
        output[o] = value;
        grad_output[o] = value - target;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float grad = 0.0f;
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            grad += grad_output[o] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
            mlp->w1[tkm_mlp_w1_offset(mlp, h, o)] -= learning_rate * grad_output[o] * hidden[h];
        }
        grad_hidden[h] = pre_hidden[h] > 0.0f ? grad : 0.0f;
        mlp->b0[h] -= learning_rate * grad_hidden[h];
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        mlp->b1[o] -= learning_rate * grad_output[o];
    }

    for (uint32_t i = 0; i < mlp->input_dim; i++) {
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            mlp->w0[tkm_mlp_w0_offset(mlp, i, h)] -= learning_rate * grad_hidden[h] * input[i];
        }
    }

    (void)output;
    return TKM_OK;
}

int tkm_mlp_window_train_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    uint16_t target_index,
    float learning_rate
) {
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_output[TKM_MLP_WINDOW_MAX_OUTPUT];
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float max_logit;
    float sum_exp = 0.0f;

    if (!tkm_mlp_ready(mlp) || !input || target_index >= mlp->output_dim || learning_rate <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float value = mlp->b0[h];
        for (uint32_t i = 0; i < mlp->input_dim; i++) {
            value += input[i] * mlp->w0[tkm_mlp_w0_offset(mlp, i, h)];
        }
        pre_hidden[h] = value;
        hidden[h] = tkm_mlp_relu(value);
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            old_w1[tkm_mlp_w1_offset(mlp, h, o)] = mlp->w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float value = mlp->b1[o];
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            value += hidden[h] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
        logits[o] = value;
    }

    max_logit = logits[0];
    for (uint32_t o = 1; o < mlp->output_dim; o++) {
        if (logits[o] > max_logit) {
            max_logit = logits[o];
        }
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float value = expf(logits[o] - max_logit);
        grad_output[o] = value;
        sum_exp += value;
    }

    if (sum_exp <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float target = o == target_index ? 1.0f : 0.0f;
        grad_output[o] = grad_output[o] / sum_exp - target;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float grad = 0.0f;
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            grad += grad_output[o] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
            mlp->w1[tkm_mlp_w1_offset(mlp, h, o)] -= learning_rate * grad_output[o] * hidden[h];
        }
        grad_hidden[h] = pre_hidden[h] > 0.0f ? grad : 0.0f;
        mlp->b0[h] -= learning_rate * grad_hidden[h];
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        mlp->b1[o] -= learning_rate * grad_output[o];
    }

    for (uint32_t i = 0; i < mlp->input_dim; i++) {
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            mlp->w0[tkm_mlp_w0_offset(mlp, i, h)] -= learning_rate * grad_hidden[h] * input[i];
        }
    }

    return TKM_OK;
}

int tkm_mlp_window_train_masked_cross_entropy(
    TkmMlpWindow* mlp,
    const float* input,
    const uint8_t* valid_mask,
    uint16_t target_index,
    float learning_rate
) {
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_output[TKM_MLP_WINDOW_MAX_OUTPUT];
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float max_logit = 0.0f;
    float sum_exp = 0.0f;
    uint8_t found_valid = 0;

    if (!tkm_mlp_ready(mlp) ||
        !input ||
        !valid_mask ||
        target_index >= mlp->output_dim ||
        !valid_mask[target_index] ||
        learning_rate <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float value = mlp->b0[h];
        for (uint32_t i = 0; i < mlp->input_dim; i++) {
            value += input[i] * mlp->w0[tkm_mlp_w0_offset(mlp, i, h)];
        }
        pre_hidden[h] = value;
        hidden[h] = tkm_mlp_relu(value);
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            old_w1[tkm_mlp_w1_offset(mlp, h, o)] = mlp->w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        float value = mlp->b1[o];
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            value += hidden[h] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
        }
        logits[o] = value;
        grad_output[o] = 0.0f;

        if (valid_mask[o] && (!found_valid || value > max_logit)) {
            found_valid = 1;
            max_logit = value;
        }
    }

    if (!found_valid) {
        return TKM_ERR;
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        if (valid_mask[o]) {
            float value = expf(logits[o] - max_logit);
            grad_output[o] = value;
            sum_exp += value;
        }
    }

    if (sum_exp <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        if (valid_mask[o]) {
            float target = o == target_index ? 1.0f : 0.0f;
            grad_output[o] = grad_output[o] / sum_exp - target;
        }
    }

    for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
        float grad = 0.0f;
        for (uint32_t o = 0; o < mlp->output_dim; o++) {
            grad += grad_output[o] * old_w1[tkm_mlp_w1_offset(mlp, h, o)];
            mlp->w1[tkm_mlp_w1_offset(mlp, h, o)] -= learning_rate * grad_output[o] * hidden[h];
        }
        grad_hidden[h] = pre_hidden[h] > 0.0f ? grad : 0.0f;
        mlp->b0[h] -= learning_rate * grad_hidden[h];
    }

    for (uint32_t o = 0; o < mlp->output_dim; o++) {
        mlp->b1[o] -= learning_rate * grad_output[o];
    }

    for (uint32_t i = 0; i < mlp->input_dim; i++) {
        for (uint32_t h = 0; h < mlp->hidden_dim; h++) {
            mlp->w0[tkm_mlp_w0_offset(mlp, i, h)] -= learning_rate * grad_hidden[h] * input[i];
        }
    }

    return TKM_OK;
}
