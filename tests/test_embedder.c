#include <stdio.h>
#include <stdlib.h>

#include "core/embedder/embedder.h"

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

static void test_lookup_embedder_copies_token_vector(void) {
    const float table[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    TkmLookupEmbedder embedder;
    float out[2] = {0.0f, 0.0f};

    CHECK(tkm_lookup_embedder_init(&embedder, 3, 2, table) == TKM_OK);
    CHECK(tkm_lookup_embedder_encode(&embedder, 1, out, 2) == TKM_OK);
    check_close(out[0], 3.0f);
    check_close(out[1], 4.0f);
}

static void test_lookup_embedder_batch_keeps_tokens_flat_and_ordered(void) {
    const float table[6] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const uint16_t tokens[2] = {2, 0};
    TkmLookupEmbedder embedder;
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    CHECK(tkm_lookup_embedder_init(&embedder, 3, 2, table) == TKM_OK);
    CHECK(tkm_lookup_embedder_encode_batch(&embedder, tokens, 2, out, 4) == TKM_OK);
    check_close(out[0], 5.0f);
    check_close(out[1], 6.0f);
    check_close(out[2], 1.0f);
    check_close(out[3], 2.0f);
}

static void test_linear_embedder_projects_continuous_vector(void) {
    const float weights[6] = {
        1.0f, 0.0f, 2.0f,
        -1.0f, 1.0f, 0.5f,
    };
    const float bias[3] = {0.5f, -0.5f, 1.0f};
    const float input[2] = {2.0f, 3.0f};
    TkmLinearEmbedder embedder;
    float out[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_linear_embedder_init(&embedder, 2, 3, weights, bias) == TKM_OK);
    CHECK(tkm_linear_embedder_encode(&embedder, input, out, 3) == TKM_OK);
    check_close(out[0], -0.5f);
    check_close(out[1], 2.5f);
    check_close(out[2], 6.5f);
}

static void test_embedder_rejects_bad_inputs(void) {
    const float table[2] = {1.0f, 2.0f};
    const uint16_t tokens[1] = {0};
    const float input[1] = {1.0f};
    TkmLookupEmbedder lookup;
    TkmLinearEmbedder linear;
    float out[2] = {0.0f, 0.0f};

    CHECK(tkm_lookup_embedder_init(0, 1, 2, table) == TKM_ERR);
    CHECK(tkm_lookup_embedder_init(&lookup, 0, 2, table) == TKM_ERR);
    CHECK(tkm_lookup_embedder_init(&lookup, 1, 0, table) == TKM_ERR);
    CHECK(tkm_lookup_embedder_init(&lookup, TKM_EMBED_MAX_TOKENS + 1, 2, table) == TKM_ERR);
    CHECK(tkm_lookup_embedder_init(&lookup, 1, TKM_EMBED_MAX_DIM + 1, table) == TKM_ERR);
    CHECK(tkm_lookup_embedder_init(&lookup, 1, 2, table) == TKM_OK);
    CHECK(tkm_lookup_embedder_encode(&lookup, 1, out, 2) == TKM_ERR);
    CHECK(tkm_lookup_embedder_encode(&lookup, 0, out, 1) == TKM_ERR);
    CHECK(tkm_lookup_embedder_encode_batch(&lookup, tokens, 0, out, 2) == TKM_ERR);

    CHECK(tkm_linear_embedder_init(0, 1, 2, table, out) == TKM_ERR);
    CHECK(tkm_linear_embedder_init(&linear, 0, 2, table, out) == TKM_ERR);
    CHECK(tkm_linear_embedder_init(&linear, 1, 0, table, out) == TKM_ERR);
    CHECK(tkm_linear_embedder_init(&linear, TKM_LINEAR_MAX_INPUT + 1, 2, table, out) == TKM_ERR);
    CHECK(tkm_linear_embedder_init(&linear, 1, TKM_EMBED_MAX_DIM + 1, table, out) == TKM_ERR);
    CHECK(tkm_linear_embedder_init(&linear, 1, 2, table, out) == TKM_OK);
    CHECK(tkm_linear_embedder_encode(&linear, 0, out, 2) == TKM_ERR);
    CHECK(tkm_linear_embedder_encode(&linear, input, out, 1) == TKM_ERR);
}

int main(void) {
    test_lookup_embedder_copies_token_vector();
    test_lookup_embedder_batch_keeps_tokens_flat_and_ordered();
    test_linear_embedder_projects_continuous_vector();
    test_embedder_rejects_bad_inputs();
    puts("embedder tests passed");
    return 0;
}
