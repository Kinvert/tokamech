#include <stdio.h>
#include <stdlib.h>

#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_lookup_policy_can_weight_sampled_returns_over_majority(void) {
    TkmTrajectory trajectory;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;

    tkm_trajectory_init(&trajectory);
    CHECK(tkm_int_bins_init(&tokenizer, -2, 2) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 1, 1, 0.0f, 0) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 1, 1, 0.0f, 1) == TKM_OK);
    CHECK(tkm_trajectory_append(&trajectory, 1, 2, 1.0f, 1) == TKM_OK);

    CHECK(tkm_lookup_policy_train_return_weighted(&policy, &tokenizer, &trajectory, 3, 1.0f) == TKM_OK);
    CHECK(tkm_lookup_policy_predict(&policy, tkm_int_bins_encode(&tokenizer, 1)) == 2);
}

int main(void) {
    test_lookup_policy_can_weight_sampled_returns_over_majority();
    puts("lookup policy tests passed");
    return 0;
}
