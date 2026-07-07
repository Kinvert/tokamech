#include <stdio.h>
#include <stdlib.h>

#include "core/head/categorical.h"

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

static void test_categorical_softmax_outputs_probabilities(void) {
    const float logits[3] = {1.0f, 2.0f, 3.0f};
    float probs[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_categorical_softmax(logits, 3, probs) == TKM_OK);
    check_close(probs[0] + probs[1] + probs[2], 1.0f);
    CHECK(probs[2] > probs[1]);
    CHECK(probs[1] > probs[0]);
}

static void test_categorical_softmax_is_stable_for_large_logits(void) {
    const float logits[2] = {1000.0f, 1001.0f};
    float probs[2] = {0.0f, 0.0f};

    CHECK(tkm_categorical_softmax(logits, 2, probs) == TKM_OK);
    check_close(probs[0] + probs[1], 1.0f);
    CHECK(probs[1] > probs[0]);
    CHECK(probs[0] > 0.0f);
}

static void test_categorical_argmax_returns_first_max_index(void) {
    const float values[4] = {1.0f, 3.0f, 3.0f, 2.0f};
    uint16_t index = 99;

    CHECK(tkm_categorical_argmax(values, 4, &index) == TKM_OK);
    CHECK(index == 1);
}

static void test_categorical_head_rejects_bad_inputs(void) {
    const float values[1] = {1.0f};
    float probs[1] = {0.0f};
    uint16_t index = 0;

    CHECK(tkm_categorical_softmax(0, 1, probs) == TKM_ERR);
    CHECK(tkm_categorical_softmax(values, 0, probs) == TKM_ERR);
    CHECK(tkm_categorical_softmax(values, 1, 0) == TKM_ERR);
    CHECK(tkm_categorical_argmax(0, 1, &index) == TKM_ERR);
    CHECK(tkm_categorical_argmax(values, 0, &index) == TKM_ERR);
    CHECK(tkm_categorical_argmax(values, 1, 0) == TKM_ERR);
}

int main(void) {
    test_categorical_softmax_outputs_probabilities();
    test_categorical_softmax_is_stable_for_large_logits();
    test_categorical_argmax_returns_first_max_index();
    test_categorical_head_rejects_bad_inputs();
    puts("categorical head tests passed");
    return 0;
}
