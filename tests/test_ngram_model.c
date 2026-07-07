#include <stdio.h>
#include <stdlib.h>

#include "core/model/ngram.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_ngram_trains_bigram_counts_and_predicts_most_common_next_token(void) {
    const uint16_t sequence[9] = {1, 2, 3, 1, 2, 4, 1, 2, 4};
    TkmNgramModel model;
    uint16_t token = 0;

    CHECK(tkm_ngram_init(&model, 8) == TKM_OK);
    CHECK(tkm_ngram_train_sequence(&model, sequence, 9) == TKM_OK);
    CHECK(tkm_ngram_predict_next(&model, 2, &token) == TKM_OK);
    CHECK(token == 4);
}

static void test_ngram_predicts_lowest_token_on_ties_and_zero_for_unknown_context(void) {
    const uint16_t sequence[5] = {1, 3, 1, 2, 4};
    TkmNgramModel model;
    uint16_t token = 99;

    CHECK(tkm_ngram_init(&model, 8) == TKM_OK);
    CHECK(tkm_ngram_train_sequence(&model, sequence, 5) == TKM_OK);
    CHECK(tkm_ngram_predict_next(&model, 1, &token) == TKM_OK);
    CHECK(token == 2);
    CHECK(tkm_ngram_predict_next(&model, 7, &token) == TKM_OK);
    CHECK(token == 0);
}

static void test_ngram_rejects_bad_inputs(void) {
    const uint16_t sequence[2] = {0, 1};
    TkmNgramModel model;
    uint16_t token = 0;

    CHECK(tkm_ngram_init(0, 8) == TKM_ERR);
    CHECK(tkm_ngram_init(&model, 0) == TKM_ERR);
    CHECK(tkm_ngram_init(&model, TKM_NGRAM_MAX_VOCAB + 1) == TKM_ERR);
    CHECK(tkm_ngram_init(&model, 8) == TKM_OK);
    CHECK(tkm_ngram_train_sequence(&model, 0, 2) == TKM_ERR);
    CHECK(tkm_ngram_train_sequence(&model, sequence, 1) == TKM_ERR);
    CHECK(tkm_ngram_train_sequence(&model, sequence, 2) == TKM_OK);
    CHECK(tkm_ngram_train_sequence(&model, (const uint16_t[]){0, 8}, 2) == TKM_ERR);
    CHECK(tkm_ngram_predict_next(&model, 8, &token) == TKM_ERR);
    CHECK(tkm_ngram_predict_next(&model, 0, 0) == TKM_ERR);
}

int main(void) {
    test_ngram_trains_bigram_counts_and_predicts_most_common_next_token();
    test_ngram_predicts_lowest_token_on_ties_and_zero_for_unknown_context();
    test_ngram_rejects_bad_inputs();
    puts("ngram model tests passed");
    return 0;
}
