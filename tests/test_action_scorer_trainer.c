#include <stdio.h>
#include <stdlib.h>

#include "core/train/action_scorer_trainer.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void init_test_scorer(TkmActionScorer* scorer) {
    const float w0[3 * 2] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float b0[2] = {0.0f, 0.0f};
    const float w1[2] = {0.2f, 0.0f};
    const float b1[1] = {0.0f};

    CHECK(tkm_action_scorer_init(scorer, 1, 2, 2, w0, b0, w1, b1) == TKM_OK);
}

static void test_configured_trainer_dispatches_masked_cross_entropy(void) {
    TkmActionScorer scorer;
    const float state[1] = {0.0f};
    const float candidates[3 * 2] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        5.0f, 0.0f,
    };
    const uint8_t mask[3] = {1, 1, 0};
    uint16_t action = 0;

    init_test_scorer(&scorer);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 3, &action) == TKM_OK);
    CHECK(action == 0);

    for (uint32_t i = 0; i < 32; i++) {
        CHECK(tkm_action_scorer_train_configured(
            &scorer,
            TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY,
            state,
            candidates,
            mask,
            3,
            1,
            0.35f,
            0.1f
        ) == TKM_OK);
    }

    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 3, &action) == TKM_OK);
    CHECK(action == 1);
}

static void test_configured_trainer_dispatches_margin(void) {
    TkmActionScorer scorer;
    const float state[1] = {0.0f};
    const float candidates[2 * 2] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const uint8_t mask[2] = {1, 1};
    uint16_t action = 0;

    init_test_scorer(&scorer);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 2, &action) == TKM_OK);
    CHECK(action == 0);

    for (uint32_t i = 0; i < 8; i++) {
        CHECK(tkm_action_scorer_train_configured(
            &scorer,
            TKM_TRAIN_LOSS_MARGIN,
            state,
            candidates,
            mask,
            2,
            1,
            0.35f,
            0.1f
        ) == TKM_OK);
    }

    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 2, &action) == TKM_OK);
    CHECK(action == 1);
}

static void test_configured_trainer_rejects_bad_inputs(void) {
    TkmActionScorer scorer;
    const float state[1] = {0.0f};
    const float action_features[2] = {1.0f, 0.0f};
    const uint8_t mask[1] = {1};
    const uint8_t invalid_mask[1] = {0};

    init_test_scorer(&scorer);

    CHECK(tkm_action_scorer_train_configured(
        0,
        TKM_TRAIN_LOSS_MARGIN,
        state,
        action_features,
        mask,
        1,
        0,
        0.35f,
        0.1f
    ) == TKM_ERR);
    CHECK(tkm_action_scorer_train_configured(
        &scorer,
        TKM_TRAIN_LOSS_CROSS_ENTROPY,
        state,
        action_features,
        mask,
        1,
        0,
        0.35f,
        0.1f
    ) == TKM_ERR);
    CHECK(tkm_action_scorer_train_configured(
        &scorer,
        TKM_TRAIN_LOSS_MARGIN,
        state,
        action_features,
        invalid_mask,
        1,
        0,
        0.35f,
        0.1f
    ) == TKM_ERR);
    CHECK(tkm_action_scorer_train_configured(
        &scorer,
        TKM_TRAIN_LOSS_MARGIN,
        state,
        action_features,
        mask,
        1,
        1,
        0.35f,
        0.1f
    ) == TKM_ERR);
}

int main(void) {
    test_configured_trainer_dispatches_masked_cross_entropy();
    test_configured_trainer_dispatches_margin();
    test_configured_trainer_rejects_bad_inputs();
    puts("action scorer trainer tests passed");
    return 0;
}
