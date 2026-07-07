#include <stdio.h>
#include <stdlib.h>

#include "core/model/action_scorer.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_action_scorer_scores_and_masks_candidate_actions(void) {
    TkmActionScorer scorer;
    const float w0[3] = {0.0f, 1.0f, 1.0f};
    const float b0[1] = {0.0f};
    const float w1[1] = {1.0f};
    const float b1[1] = {0.0f};
    const float state[1] = {0.0f};
    const float candidates[3 * 2] = {
        1.0f, 0.0f,
        0.0f, 3.0f,
        0.0f, 2.0f,
    };
    const uint8_t mask[3] = {1, 0, 1};
    float scores[3] = {0.0f, 0.0f, 0.0f};
    uint16_t action = 0;

    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 1, w0, b0, w1, b1) == TKM_OK);
    CHECK(tkm_action_scorer_score_all(&scorer, state, candidates, 3, scores, 3) == TKM_OK);
    CHECK(scores[1] > scores[2]);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 3, &action) == TKM_OK);
    CHECK(action == 2);
}

static void test_action_scorer_margin_update_raises_expert_score(void) {
    TkmActionScorer scorer;
    const float w0[3] = {0.0f, 1.0f, 1.0f};
    const float b0[1] = {0.0f};
    const float w1[1] = {1.0f};
    const float b1[1] = {0.0f};
    const float state[1] = {0.0f};
    const float expert[2] = {1.0f, 0.0f};
    const float other[2] = {0.0f, 1.0f};
    float before_expert = 0.0f;
    float before_other = 0.0f;
    float after_expert = 0.0f;
    float after_other = 0.0f;

    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 1, w0, b0, w1, b1) == TKM_OK);
    CHECK(tkm_action_scorer_score(&scorer, state, expert, &before_expert) == TKM_OK);
    CHECK(tkm_action_scorer_score(&scorer, state, other, &before_other) == TKM_OK);
    CHECK(before_expert == before_other);
    CHECK(tkm_action_scorer_train_margin(&scorer, state, expert, other, 1.0f, 0.1f) == TKM_OK);
    CHECK(tkm_action_scorer_score(&scorer, state, expert, &after_expert) == TKM_OK);
    CHECK(tkm_action_scorer_score(&scorer, state, other, &after_other) == TKM_OK);
    CHECK(after_expert > before_expert);
    CHECK(after_other < before_other);
    CHECK(after_expert > after_other);
}

static void test_action_scorer_masked_cross_entropy_learns_target_action(void) {
    TkmActionScorer scorer;
    const float w0[3 * 2] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float b0[2] = {0.0f, 0.0f};
    const float w1[2] = {0.2f, 0.0f};
    const float b1[1] = {0.0f};
    const float state[1] = {0.0f};
    const float candidates[3 * 2] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        5.0f, 0.0f,
    };
    const uint8_t mask[3] = {1, 1, 0};
    float before_scores[3] = {0.0f, 0.0f, 0.0f};
    float after_scores[3] = {0.0f, 0.0f, 0.0f};
    uint16_t action = 0;

    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 2, w0, b0, w1, b1) == TKM_OK);
    CHECK(tkm_action_scorer_score_all(&scorer, state, candidates, 3, before_scores, 3) == TKM_OK);
    CHECK(before_scores[2] > before_scores[0]);
    CHECK(before_scores[0] > before_scores[1]);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 3, &action) == TKM_OK);
    CHECK(action == 0);

    for (uint32_t i = 0; i < 32; i++) {
        CHECK(tkm_action_scorer_train_masked_cross_entropy(
            &scorer,
            state,
            candidates,
            mask,
            3,
            1,
            0.1f
        ) == TKM_OK);
    }

    CHECK(tkm_action_scorer_score_all(&scorer, state, candidates, 3, after_scores, 3) == TKM_OK);
    CHECK(after_scores[1] > after_scores[0]);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, candidates, mask, 3, &action) == TKM_OK);
    CHECK(action == 1);
}

static void test_action_scorer_rejects_bad_inputs(void) {
    TkmActionScorer scorer;
    const float weights[3] = {1.0f, 1.0f, 1.0f};
    const float state[1] = {0.0f};
    const float action[2] = {1.0f, 0.0f};
    const uint8_t mask[1] = {1};
    const uint8_t invalid_mask[1] = {0};
    float score = 0.0f;
    uint16_t index = 0;

    CHECK(tkm_action_scorer_init(0, 1, 2, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 0, 2, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 1, 0, 1, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 0, weights, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 1, 0, 0, weights, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 1, weights, 0, 0, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_init(&scorer, 1, 2, 1, weights, 0, weights, 0) == TKM_OK);
    CHECK(tkm_action_scorer_score(0, state, action, &score) == TKM_ERR);
    CHECK(tkm_action_scorer_score(&scorer, 0, action, &score) == TKM_ERR);
    CHECK(tkm_action_scorer_score(&scorer, state, 0, &score) == TKM_ERR);
    CHECK(tkm_action_scorer_score(&scorer, state, action, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, action, 0, 1, &index) == TKM_ERR);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, action, mask, 0, &index) == TKM_ERR);
    CHECK(tkm_action_scorer_masked_argmax(&scorer, state, action, mask, 1, 0) == TKM_ERR);
    CHECK(tkm_action_scorer_train_margin(&scorer, state, action, action, 1.0f, 0.0f) == TKM_ERR);
    CHECK(tkm_action_scorer_train_masked_cross_entropy(&scorer, state, action, mask, 1, 0, 0.0f) == TKM_ERR);
    CHECK(tkm_action_scorer_train_masked_cross_entropy(&scorer, state, action, invalid_mask, 1, 0, 0.1f) == TKM_ERR);
    CHECK(tkm_action_scorer_train_masked_cross_entropy(&scorer, state, action, mask, 1, 1, 0.1f) == TKM_ERR);
}

int main(void) {
    test_action_scorer_scores_and_masks_candidate_actions();
    test_action_scorer_margin_update_raises_expert_score();
    test_action_scorer_masked_cross_entropy_learns_target_action();
    test_action_scorer_rejects_bad_inputs();
    puts("action scorer tests passed");
    return 0;
}
