#include <stdio.h>
#include <stdlib.h>

#include "core/tokenizer/vq_code.h"

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

static void test_vq_code_encodes_nearest_fixed_codebook_vector(void) {
    const float codebook[6] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float vector[2] = {0.9f, 0.1f};
    TkmVqCode vq;
    uint16_t code = 0;
    float distance = 0.0f;

    CHECK(tkm_vq_code_init(&vq, 3, 2, codebook) == TKM_OK);
    CHECK(tkm_vq_code_encode(&vq, vector, &code, &distance) == TKM_OK);
    CHECK(code == 1);
    check_close(distance, 0.02f);
}

static void test_vq_code_decode_copies_code_vector(void) {
    const float codebook[6] = {
        -1.0f, -1.0f,
        2.0f, 3.0f,
        4.0f, 5.0f,
    };
    TkmVqCode vq;
    float decoded[2] = {0.0f, 0.0f};

    CHECK(tkm_vq_code_init(&vq, 3, 2, codebook) == TKM_OK);
    CHECK(tkm_vq_code_decode(&vq, 2, decoded, 2) == TKM_OK);
    check_close(decoded[0], 4.0f);
    check_close(decoded[1], 5.0f);
}

static void test_vq_code_batch_encode_keeps_vectors_flat_and_explicit(void) {
    const float codebook[6] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float vectors[6] = {
        0.1f, 0.0f,
        0.8f, 0.2f,
        0.1f, 0.9f,
    };
    TkmVqCode vq;
    uint16_t codes[3] = {99, 99, 99};

    CHECK(tkm_vq_code_init(&vq, 3, 2, codebook) == TKM_OK);
    CHECK(tkm_vq_code_encode_batch(&vq, vectors, 3, codes) == TKM_OK);
    CHECK(codes[0] == 0);
    CHECK(codes[1] == 1);
    CHECK(codes[2] == 2);
}

static void test_vq_code_rejects_bad_inputs(void) {
    const float codebook[2] = {0.0f, 0.0f};
    const float vector[2] = {0.0f, 0.0f};
    TkmVqCode vq;
    uint16_t code = 0;
    float distance = 0.0f;
    float decoded[2] = {0.0f, 0.0f};

    CHECK(tkm_vq_code_init(0, 1, 2, codebook) == TKM_ERR);
    CHECK(tkm_vq_code_init(&vq, 0, 2, codebook) == TKM_ERR);
    CHECK(tkm_vq_code_init(&vq, 1, 0, codebook) == TKM_ERR);
    CHECK(tkm_vq_code_init(&vq, TKM_VQ_MAX_CODES + 1, 2, codebook) == TKM_ERR);
    CHECK(tkm_vq_code_init(&vq, 1, TKM_VQ_MAX_DIM + 1, codebook) == TKM_ERR);
    CHECK(tkm_vq_code_init(&vq, 1, 2, codebook) == TKM_OK);
    CHECK(tkm_vq_code_encode(&vq, 0, &code, &distance) == TKM_ERR);
    CHECK(tkm_vq_code_encode(&vq, vector, 0, &distance) == TKM_ERR);
    CHECK(tkm_vq_code_decode(&vq, 1, decoded, 2) == TKM_ERR);
    CHECK(tkm_vq_code_decode(&vq, 0, decoded, 1) == TKM_ERR);
    CHECK(tkm_vq_code_encode_batch(&vq, vector, 0, &code) == TKM_ERR);
}

int main(void) {
    test_vq_code_encodes_nearest_fixed_codebook_vector();
    test_vq_code_decode_copies_code_vector();
    test_vq_code_batch_encode_keeps_vectors_flat_and_explicit();
    test_vq_code_rejects_bad_inputs();
    puts("vq code tests passed");
    return 0;
}
