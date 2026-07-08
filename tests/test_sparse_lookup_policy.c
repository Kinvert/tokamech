#include <stdio.h>
#include <stdlib.h>

#include "core/dataset/trajectory.h"
#include "core/train/sparse_lookup_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_sparse_lookup_learns_large_tokens_by_majority_vote(void) {
    TkmTrajectory trajectory;
    TkmSparseLookupPolicy policy;

    tkm_trajectory_init(&trajectory);
    CHECK(tkm_trajectory_append(&trajectory, 1234567, 2, 0.0f, 0) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 1234567, 3, 0.0f, 0) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 1234567, 3, 0.0f, 0) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 7654321, 1, 0.0f, 0) == TKM_OK);

    CHECK(tkm_sparse_lookup_policy_train(&policy, &trajectory, 4, 0) == TKM_OK);
    CHECK(policy.item_count == 2);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, 1234567) == 3);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, 7654321) == 1);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, 1111111) == 0);
}

static void test_sparse_lookup_can_weight_sampled_returns_over_majority(void) {
    TkmTrajectory trajectory;
    TkmSparseLookupPolicy policy;

    tkm_trajectory_init(&trajectory);
    CHECK(tkm_trajectory_append(&trajectory, 42, 1, 0.0f, 0) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 42, 1, 0.0f, 1) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 42, 2, 1.0f, 1) == TKM_OK);

    CHECK(tkm_sparse_lookup_policy_train_return_weighted(&policy, &trajectory, 3, 0, 1.0f) == TKM_OK);
    CHECK(tkm_sparse_lookup_policy_predict(&policy, 42) == 2);
}

int main(void) {
    test_sparse_lookup_learns_large_tokens_by_majority_vote();
    test_sparse_lookup_can_weight_sampled_returns_over_majority();
    puts("sparse lookup policy tests passed");
    return 0;
}
