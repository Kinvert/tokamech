#include <stdio.h>
#include <stdlib.h>

#include "core/model/mlp_window.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void check_close(float actual, float expected) {
    float diff = actual - expected;
    if (diff < 0.0f) {
        diff = -diff;
    }
    CHECK(diff < 0.0001f);
}

static void test_mlp_window_forward_runs_one_hidden_relu_layer(void) {
    const float w0[6] = {
        1.0f, -1.0f, 0.5f,
        0.0f, 2.0f, -1.0f,
    };
    const float b0[3] = {0.0f, -1.0f, 0.5f};
    const float w1[6] = {
        1.0f, 0.0f,
        0.5f, 1.0f,
        -1.0f, 2.0f,
    };
    const float b1[2] = {0.25f, -0.25f};
    const float input[2] = {2.0f, 3.0f};
    TkmMlpWindow mlp;
    float output[2] = {0.0f, 0.0f};

    CHECK(tkm_mlp_window_init(&mlp, 2, 3, 2, w0, b0, w1, b1) == TKM_OK);
    CHECK(tkm_mlp_window_forward(&mlp, input, output, 2) == TKM_OK);
    check_close(output[0], 3.75f);
    check_close(output[1], 2.75f);
}

static void test_mlp_window_supports_zero_bias_pointers(void) {
    const float w0[2] = {1.0f, -2.0f};
    const float w1[2] = {3.0f, 4.0f};
    const float input[1] = {2.0f};
    TkmMlpWindow mlp;
    float output[2] = {0.0f, 0.0f};

    CHECK(tkm_mlp_window_init(&mlp, 1, 1, 2, w0, 0, w1, 0) == TKM_OK);
    CHECK(tkm_mlp_window_forward(&mlp, input, output, 2) == TKM_OK);
    check_close(output[0], 6.0f);
    check_close(output[1], 8.0f);
}

static uint16_t test_argmax(const float* values, uint32_t count) {
    uint16_t best = 0;
    for (uint32_t i = 1; i < count; i++) {
        if (values[i] > values[best]) {
            best = (uint16_t)i;
        }
    }
    return best;
}

static uint16_t test_masked_argmax(const float* values, const uint8_t* mask, uint32_t count) {
    uint16_t best = 0;
    uint8_t found = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!mask[i]) {
            continue;
        }
        if (!found || values[i] > values[best]) {
            best = (uint16_t)i;
            found = 1;
        }
    }
    CHECK(found);
    return best;
}

static void test_mlp_window_trains_output_classifier_with_mse(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    const float neg[1] = {-1.0f};
    const float zero[1] = {0.0f};
    const float pos[1] = {1.0f};
    TkmMlpWindow mlp;
    float logits[3];

    CHECK(tkm_mlp_window_init(&mlp, 1, 3, 3, w0, b0, w1, b1) == TKM_OK);
    for (uint32_t epoch = 0; epoch < 40; epoch++) {
        CHECK(tkm_mlp_window_train_mse(&mlp, neg, 0, 0.05f) == TKM_OK);
        CHECK(tkm_mlp_window_train_mse(&mlp, zero, 1, 0.05f) == TKM_OK);
        CHECK(tkm_mlp_window_train_mse(&mlp, pos, 2, 0.05f) == TKM_OK);
    }

    CHECK(tkm_mlp_window_forward(&mlp, neg, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 0);
    CHECK(tkm_mlp_window_forward(&mlp, zero, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 1);
    CHECK(tkm_mlp_window_forward(&mlp, pos, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 2);
}

static void test_mlp_window_trains_output_classifier_with_cross_entropy(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    const float neg[1] = {-1.0f};
    const float zero[1] = {0.0f};
    const float pos[1] = {1.0f};
    TkmMlpWindow mlp;
    float logits[3];

    CHECK(tkm_mlp_window_init(&mlp, 1, 3, 3, w0, b0, w1, b1) == TKM_OK);
    for (uint32_t epoch = 0; epoch < 40; epoch++) {
        CHECK(tkm_mlp_window_train_cross_entropy(&mlp, neg, 0, 0.08f) == TKM_OK);
        CHECK(tkm_mlp_window_train_cross_entropy(&mlp, zero, 1, 0.08f) == TKM_OK);
        CHECK(tkm_mlp_window_train_cross_entropy(&mlp, pos, 2, 0.08f) == TKM_OK);
    }

    CHECK(tkm_mlp_window_forward(&mlp, neg, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 0);
    CHECK(tkm_mlp_window_forward(&mlp, zero, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 1);
    CHECK(tkm_mlp_window_forward(&mlp, pos, logits, 3) == TKM_OK);
    CHECK(test_argmax(logits, 3) == 2);
}

static void test_mlp_window_trains_output_classifier_with_masked_cross_entropy(void) {
    const float w0[3] = {1.0f, -1.0f, 0.0f};
    const float b0[3] = {0.0f, 0.0f, 1.0f};
    const float w1[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const float b1[3] = {0.0f, 0.0f, 0.0f};
    const float neg[1] = {-1.0f};
    const float zero[1] = {0.0f};
    const float pos[1] = {1.0f};
    const uint8_t neg_mask[3] = {1, 0, 1};
    const uint8_t zero_mask[3] = {0, 1, 1};
    const uint8_t pos_mask[3] = {1, 1, 1};
    TkmMlpWindow mlp;
    float logits[3];

    CHECK(tkm_mlp_window_init(&mlp, 1, 3, 3, w0, b0, w1, b1) == TKM_OK);
    for (uint32_t epoch = 0; epoch < 50; epoch++) {
        CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, neg, neg_mask, 0, 0.08f) == TKM_OK);
        CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, zero, zero_mask, 1, 0.08f) == TKM_OK);
        CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, pos, pos_mask, 2, 0.08f) == TKM_OK);
    }

    CHECK(tkm_mlp_window_forward(&mlp, neg, logits, 3) == TKM_OK);
    CHECK(test_masked_argmax(logits, neg_mask, 3) == 0);
    CHECK(tkm_mlp_window_forward(&mlp, zero, logits, 3) == TKM_OK);
    CHECK(test_masked_argmax(logits, zero_mask, 3) == 1);
    CHECK(tkm_mlp_window_forward(&mlp, pos, logits, 3) == TKM_OK);
    CHECK(test_masked_argmax(logits, pos_mask, 3) == 2);
}

static void test_mlp_window_rejects_bad_inputs(void) {
    const float weights[1] = {1.0f};
    const float input[1] = {1.0f};
    TkmMlpWindow mlp;
    float output[1] = {0.0f};

    CHECK(tkm_mlp_window_init(0, 1, 1, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 0, 1, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 0, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 1, 0, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, TKM_MLP_WINDOW_MAX_INPUT + 1, 1, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, TKM_MLP_WINDOW_MAX_HIDDEN + 1, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 1, TKM_MLP_WINDOW_MAX_OUTPUT + 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 1, 1, 0, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 1, 1, weights, 0, 0, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_init(&mlp, 1, 1, 1, weights, 0, weights, 0) == TKM_OK);
    CHECK(tkm_mlp_window_forward(&mlp, 0, output, 1) == TKM_ERR);
    CHECK(tkm_mlp_window_forward(&mlp, input, 0, 1) == TKM_ERR);
    CHECK(tkm_mlp_window_forward(&mlp, input, output, 0) == TKM_ERR);
    CHECK(tkm_mlp_window_train_mse(0, input, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_mse(&mlp, 0, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_mse(&mlp, input, 1, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_mse(&mlp, input, 0, 0.0f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_cross_entropy(0, input, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_cross_entropy(&mlp, 0, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_cross_entropy(&mlp, input, 1, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_cross_entropy(&mlp, input, 0, 0.0f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(0, input, (uint8_t[]){1}, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, 0, (uint8_t[]){1}, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, input, 0, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, input, (uint8_t[]){1}, 1, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, input, (uint8_t[]){0}, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_mlp_window_train_masked_cross_entropy(&mlp, input, (uint8_t[]){1}, 0, 0.0f) == TKM_ERR);
}

int main(void) {
    test_mlp_window_forward_runs_one_hidden_relu_layer();
    test_mlp_window_supports_zero_bias_pointers();
    test_mlp_window_trains_output_classifier_with_mse();
    test_mlp_window_trains_output_classifier_with_cross_entropy();
    test_mlp_window_trains_output_classifier_with_masked_cross_entropy();
    test_mlp_window_rejects_bad_inputs();
    puts("mlp window tests passed");
    return 0;
}
