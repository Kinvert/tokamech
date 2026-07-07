#include <stdio.h>
#include <stdlib.h>

#include "core/model/linear_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_linear_policy_trains_binary_separator(void) {
    TkmLinearPolicy policy;
    const float neg[1] = {-1.0f};
    const float pos[1] = {1.0f};
    uint8_t action = 99;

    CHECK(tkm_linear_policy_init(&policy, 1, 2) == TKM_OK);

    for (uint32_t epoch = 0; epoch < 8; epoch++) {
        CHECK(tkm_linear_policy_train_perceptron(&policy, neg, 0, 0.25f) == TKM_OK);
        CHECK(tkm_linear_policy_train_perceptron(&policy, pos, 1, 0.25f) == TKM_OK);
    }

    CHECK(tkm_linear_policy_predict(&policy, neg, &action) == TKM_OK);
    CHECK(action == 0);
    CHECK(tkm_linear_policy_predict(&policy, pos, &action) == TKM_OK);
    CHECK(action == 1);
}

static void test_linear_policy_logits_and_bad_inputs(void) {
    TkmLinearPolicy policy;
    const float input[1] = {1.0f};
    float logits[2] = {0.0f, 0.0f};
    uint8_t action = 99;

    CHECK(tkm_linear_policy_init(0, 1, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_init(&policy, 0, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_init(&policy, TKM_LINEAR_POLICY_MAX_INPUT + 1u, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_init(&policy, 1, 0) == TKM_ERR);
    CHECK(tkm_linear_policy_init(&policy, 1, TKM_LINEAR_POLICY_MAX_ACTIONS + 1u) == TKM_ERR);
    CHECK(tkm_linear_policy_init(&policy, 1, 2) == TKM_OK);
    CHECK(tkm_linear_policy_logits(0, input, logits, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_logits(&policy, 0, logits, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_logits(&policy, input, 0, 2) == TKM_ERR);
    CHECK(tkm_linear_policy_logits(&policy, input, logits, 1) == TKM_ERR);
    CHECK(tkm_linear_policy_predict(0, input, &action) == TKM_ERR);
    CHECK(tkm_linear_policy_predict(&policy, 0, &action) == TKM_ERR);
    CHECK(tkm_linear_policy_predict(&policy, input, 0) == TKM_ERR);
    CHECK(tkm_linear_policy_train_perceptron(0, input, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_linear_policy_train_perceptron(&policy, 0, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_linear_policy_train_perceptron(&policy, input, 2, 0.1f) == TKM_ERR);
    CHECK(tkm_linear_policy_train_perceptron(&policy, input, 0, 0.0f) == TKM_ERR);
    CHECK(tkm_linear_policy_logits(&policy, input, logits, 2) == TKM_OK);
}

int main(void) {
    test_linear_policy_trains_binary_separator();
    test_linear_policy_logits_and_bad_inputs();
    puts("linear policy tests passed");
    return 0;
}
