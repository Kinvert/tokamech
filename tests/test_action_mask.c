#include <stdio.h>
#include <stdlib.h>

#include "core/head/action_mask.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_action_mask_argmax_ignores_invalid_high_logits(void) {
    const float logits[4] = {0.5f, 99.0f, 2.0f, 1.0f};
    const uint8_t mask[4] = {1, 0, 1, 1};
    uint16_t action = 0;

    CHECK(tkm_action_mask_argmax(logits, mask, 4, &action) == TKM_OK);
    CHECK(action == 2);
}

static void test_action_mask_rejects_empty_masks_and_bad_inputs(void) {
    const float logits[3] = {3.0f, 2.0f, 1.0f};
    const uint8_t none_valid[3] = {0, 0, 0};
    const uint8_t one_valid[3] = {0, 1, 0};
    uint16_t action = 0;

    CHECK(tkm_action_mask_argmax(logits, none_valid, 3, &action) == TKM_ERR);
    CHECK(tkm_action_mask_argmax(0, one_valid, 3, &action) == TKM_ERR);
    CHECK(tkm_action_mask_argmax(logits, 0, 3, &action) == TKM_ERR);
    CHECK(tkm_action_mask_argmax(logits, one_valid, 0, &action) == TKM_ERR);
    CHECK(tkm_action_mask_argmax(logits, one_valid, 3, 0) == TKM_ERR);
}

int main(void) {
    test_action_mask_argmax_ignores_invalid_high_logits();
    test_action_mask_rejects_empty_masks_and_bad_inputs();
    puts("action mask tests passed");
    return 0;
}
