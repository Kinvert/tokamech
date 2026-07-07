#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/loss/loss.h"

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

static void test_cross_entropy_reads_target_probability(void) {
    const float probs[3] = {0.1f, 0.7f, 0.2f};
    float loss = 0.0f;

    CHECK(tkm_loss_cross_entropy(probs, 3, 1, &loss) == TKM_OK);
    check_close(loss, -logf(0.7f));
}

static void test_mse_averages_squared_error(void) {
    const float predicted[3] = {1.0f, 2.0f, 4.0f};
    const float target[3] = {1.0f, 1.0f, 1.0f};
    float loss = 0.0f;

    CHECK(tkm_loss_mse(predicted, target, 3, &loss) == TKM_OK);
    check_close(loss, 10.0f / 3.0f);
}

static void test_huber_is_quadratic_near_zero_and_linear_for_large_error(void) {
    const float predicted[3] = {1.5f, 3.0f, -2.0f};
    const float target[3] = {1.0f, 1.0f, 0.0f};
    float loss = 0.0f;

    CHECK(tkm_loss_huber(predicted, target, 3, 1.0f, &loss) == TKM_OK);
    check_close(loss, (0.125f + 1.5f + 1.5f) / 3.0f);
}

static void test_binary_cross_entropy_handles_terminal_style_targets(void) {
    float loss = 0.0f;

    CHECK(tkm_loss_binary_cross_entropy(0.8f, 1, &loss) == TKM_OK);
    check_close(loss, -logf(0.8f));

    CHECK(tkm_loss_binary_cross_entropy(0.2f, 0, &loss) == TKM_OK);
    check_close(loss, -logf(0.8f));
}

static void test_weighted_sum_combines_named_loss_terms(void) {
    const float losses[3] = {2.0f, 5.0f, 10.0f};
    const float weights[3] = {1.0f, 0.5f, 0.1f};
    float total = 0.0f;

    CHECK(tkm_loss_weighted_sum(losses, weights, 3, &total) == TKM_OK);
    check_close(total, 5.5f);
}

static void test_action_smoothness_penalizes_action_changes(void) {
    const float actions[6] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 2.0f,
    };
    float loss = 0.0f;

    CHECK(tkm_loss_action_smoothness(actions, 3, 2, &loss) == TKM_OK);
    check_close(loss, 2.5f);
}

static void test_losses_reject_bad_inputs(void) {
    const float values[2] = {0.5f, 0.5f};
    float loss = 0.0f;

    CHECK(tkm_loss_cross_entropy(values, 2, 2, &loss) == TKM_ERR);
    CHECK(tkm_loss_mse(values, values, 0, &loss) == TKM_ERR);
    CHECK(tkm_loss_huber(values, values, 2, 0.0f, &loss) == TKM_ERR);
    CHECK(tkm_loss_binary_cross_entropy(0.5f, 2, &loss) == TKM_ERR);
    CHECK(tkm_loss_weighted_sum(values, values, 0, &loss) == TKM_ERR);
    CHECK(tkm_loss_action_smoothness(values, 1, 2, &loss) == TKM_ERR);
    CHECK(tkm_loss_action_smoothness(values, 2, 0, &loss) == TKM_ERR);
    CHECK(tkm_loss_action_smoothness(values, 2, 1, 0) == TKM_ERR);
}

int main(void) {
    test_cross_entropy_reads_target_probability();
    test_mse_averages_squared_error();
    test_huber_is_quadratic_near_zero_and_linear_for_large_error();
    test_binary_cross_entropy_handles_terminal_style_targets();
    test_weighted_sum_combines_named_loss_terms();
    test_action_smoothness_penalizes_action_changes();
    test_losses_reject_bad_inputs();
    puts("loss tests passed");
    return 0;
}
