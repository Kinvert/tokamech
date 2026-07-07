#include <stdio.h>
#include <stdlib.h>

#include "core/model/nearest_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_nearest_policy_predicts_action_from_closest_feature(void) {
    TkmNearestPolicy policy;
    const float left[2] = {-1.0f, 0.0f};
    const float right[2] = {1.0f, 0.0f};
    const float query[2] = {0.8f, 0.1f};
    uint8_t action = 99;

    CHECK(tkm_nearest_policy_init(&policy, 2, 4, 0) == TKM_OK);
    CHECK(tkm_nearest_policy_add(&policy, left, 1) == TKM_OK);
    CHECK(tkm_nearest_policy_add(&policy, right, 2) == TKM_OK);
    CHECK(tkm_nearest_policy_predict(&policy, query, &action) == TKM_OK);
    CHECK(action == 2);
}

static void test_nearest_policy_rejects_bad_inputs(void) {
    TkmNearestPolicy policy;
    const float feature[1] = {0.0f};
    uint8_t action = 99;

    CHECK(tkm_nearest_policy_init(0, 1, 2, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, 0, 2, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, TKM_NEAREST_POLICY_MAX_DIM + 1u, 2, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, 1, 0, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, 1, TKM_NEAREST_POLICY_MAX_ACTIONS + 1u, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, 1, 2, 2) == TKM_ERR);
    CHECK(tkm_nearest_policy_init(&policy, 1, 2, 0) == TKM_OK);
    CHECK(tkm_nearest_policy_add(0, feature, 1) == TKM_ERR);
    CHECK(tkm_nearest_policy_add(&policy, 0, 1) == TKM_ERR);
    CHECK(tkm_nearest_policy_add(&policy, feature, 2) == TKM_ERR);
    CHECK(tkm_nearest_policy_predict(0, feature, &action) == TKM_ERR);
    CHECK(tkm_nearest_policy_predict(&policy, 0, &action) == TKM_ERR);
    CHECK(tkm_nearest_policy_predict(&policy, feature, 0) == TKM_ERR);
    CHECK(tkm_nearest_policy_predict(&policy, feature, &action) == TKM_OK);
    CHECK(action == 0);
}

int main(void) {
    test_nearest_policy_predicts_action_from_closest_feature();
    test_nearest_policy_rejects_bad_inputs();
    puts("nearest policy tests passed");
    return 0;
}
