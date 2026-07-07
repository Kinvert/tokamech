#include <stdio.h>
#include <stdlib.h>

#include "core/params/params.h"

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

static void test_params_adds_and_retrieves_named_float_arrays(void) {
    const float values[3] = {1.0f, -2.0f, 3.5f};
    TkmParams params;
    const float* out = 0;
    uint32_t count = 0;

    tkm_params_init(&params);
    CHECK(tkm_params_add_array(&params, "weights", values, 3) == TKM_OK);
    CHECK(tkm_params_get_array(&params, "weights", &out, &count) == TKM_OK);
    CHECK(count == 3);
    check_close(out[0], 1.0f);
    check_close(out[1], -2.0f);
    check_close(out[2], 3.5f);
}

static void test_params_parse_text_supports_comments_and_commas(void) {
    const char* text =
        "# flat parameter text\n"
        "codebook = 0.0, 1.0, -2.5\n"
        "bias = 3.0 4.0 ; trailing comment\n";
    TkmParams params;
    const float* codebook = 0;
    const float* bias = 0;
    uint32_t count = 0;

    CHECK(tkm_params_parse_text(&params, text) == TKM_OK);
    CHECK(tkm_params_get_array(&params, "codebook", &codebook, &count) == TKM_OK);
    CHECK(count == 3);
    check_close(codebook[0], 0.0f);
    check_close(codebook[1], 1.0f);
    check_close(codebook[2], -2.5f);
    CHECK(tkm_params_get_array(&params, "bias", &bias, &count) == TKM_OK);
    CHECK(count == 2);
    check_close(bias[0], 3.0f);
    check_close(bias[1], 4.0f);
}

static void test_params_rejects_bad_inputs(void) {
    const float value[1] = {1.0f};
    TkmParams params;
    const float* out = 0;
    uint32_t count = 0;

    tkm_params_init(&params);
    CHECK(tkm_params_add_array(0, "x", value, 1) == TKM_ERR);
    CHECK(tkm_params_add_array(&params, 0, value, 1) == TKM_ERR);
    CHECK(tkm_params_add_array(&params, "x", 0, 1) == TKM_ERR);
    CHECK(tkm_params_add_array(&params, "x", value, 0) == TKM_ERR);
    CHECK(tkm_params_add_array(&params, "x", value, 1) == TKM_OK);
    CHECK(tkm_params_add_array(&params, "x", value, 1) == TKM_ERR);
    CHECK(tkm_params_get_array(&params, "missing", &out, &count) == TKM_ERR);
    CHECK(tkm_params_get_array(&params, "x", 0, &count) == TKM_ERR);
    CHECK(tkm_params_get_array(&params, "x", &out, 0) == TKM_ERR);
    CHECK(tkm_params_parse_text(0, "x = 1.0\n") == TKM_ERR);
    CHECK(tkm_params_parse_text(&params, 0) == TKM_ERR);
    CHECK(tkm_params_parse_text(&params, "x without equals\n") == TKM_ERR);
    CHECK(tkm_params_parse_text(&params, "x = nope\n") == TKM_ERR);
}

int main(void) {
    test_params_adds_and_retrieves_named_float_arrays();
    test_params_parse_text_supports_comments_and_commas();
    test_params_rejects_bad_inputs();
    puts("params tests passed");
    return 0;
}
