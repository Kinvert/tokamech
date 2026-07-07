#include <stdio.h>
#include <stdlib.h>

#include "core/vectorizer/continuous.h"

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

static void test_continuous_vectorizer_raw_copies_flat_vectors(void) {
    const float input[3] = {1.0f, -2.0f, 3.5f};
    TkmContinuousVectorizer vectorizer;
    float output[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_continuous_vectorizer_init_raw(&vectorizer, 3) == TKM_OK);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, output, 3) == TKM_OK);
    check_close(output[0], 1.0f);
    check_close(output[1], -2.0f);
    check_close(output[2], 3.5f);
}

static void test_continuous_vectorizer_standardizes_per_dimension(void) {
    const float mean[3] = {10.0f, 0.0f, -2.0f};
    const float std[3] = {2.0f, 4.0f, 0.5f};
    const float input[3] = {14.0f, -4.0f, -1.0f};
    TkmContinuousVectorizer vectorizer;
    float output[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_continuous_vectorizer_init_standardize(&vectorizer, 3, mean, std) == TKM_OK);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, output, 3) == TKM_OK);
    check_close(output[0], 2.0f);
    check_close(output[1], -1.0f);
    check_close(output[2], 2.0f);
}

static void test_continuous_vectorizer_rejects_bad_inputs(void) {
    const float mean[1] = {0.0f};
    const float std_good[1] = {1.0f};
    const float std_bad[1] = {0.0f};
    const float input[1] = {1.0f};
    TkmContinuousVectorizer vectorizer;
    float output[1] = {0.0f};

    CHECK(tkm_continuous_vectorizer_init_raw(0, 1) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_raw(&vectorizer, 0) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_raw(&vectorizer, TKM_CONTINUOUS_MAX_DIM + 1) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_standardize(0, 1, mean, std_good) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_standardize(&vectorizer, 0, mean, std_good) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_standardize(&vectorizer, 1, 0, std_good) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_standardize(&vectorizer, 1, mean, 0) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_standardize(&vectorizer, 1, mean, std_bad) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_init_raw(&vectorizer, 1) == TKM_OK);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, 0, output, 1) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, 0, 1) == TKM_ERR);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, output, 0) == TKM_ERR);
}

int main(void) {
    test_continuous_vectorizer_raw_copies_flat_vectors();
    test_continuous_vectorizer_standardizes_per_dimension();
    test_continuous_vectorizer_rejects_bad_inputs();
    puts("continuous vectorizer tests passed");
    return 0;
}
