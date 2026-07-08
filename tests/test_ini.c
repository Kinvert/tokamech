#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config/ini.h"

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

static void test_ini_parses_sections_keys_values_and_comments(void) {
    const char* text =
        "# tokenizer choice\n"
        "[tokenizer]\n"
        "kind = vq_code\n"
        "dim = 4\n"
        "\n"
        "; loss choice\n"
        "[loss]\n"
        "kind = huber\n"
        "delta = 1.0\n";
    TkmIni ini;
    int32_t dim = 0;
    float delta = 0.0f;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(strcmp(tkm_ini_get(&ini, "tokenizer", "kind", ""), "vq_code") == 0);
    CHECK(strcmp(tkm_ini_get(&ini, "loss", "kind", ""), "huber") == 0);
    CHECK(tkm_ini_get_i32(&ini, "tokenizer", "dim", -1, &dim) == TKM_OK);
    CHECK(dim == 4);
    CHECK(tkm_ini_get_f32(&ini, "loss", "delta", -1.0f, &delta) == TKM_OK);
    check_close(delta, 1.0f);
}

static void test_ini_uses_defaults_and_later_entries_override_earlier_entries(void) {
    const char* text =
        "[model]\n"
        "kind = lookup_policy\n"
        "kind = ngram\n";
    TkmIni ini;
    int32_t missing_i32 = 0;
    float missing_f32 = 0.0f;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(strcmp(tkm_ini_get(&ini, "model", "kind", ""), "ngram") == 0);
    CHECK(strcmp(tkm_ini_get(&ini, "model", "missing", "fallback"), "fallback") == 0);
    CHECK(tkm_ini_get_i32(&ini, "model", "missing_i32", 7, &missing_i32) == TKM_OK);
    CHECK(missing_i32 == 7);
    CHECK(tkm_ini_get_f32(&ini, "model", "missing_f32", 0.25f, &missing_f32) == TKM_OK);
    check_close(missing_f32, 0.25f);
}

static void test_ini_parse_file_loads_project_configs(void) {
    const char* path = "build/test_ini_parse_file.ini";
    FILE* file = fopen(path, "wb");
    TkmIni ini;

    CHECK(file != 0);
    CHECK(fputs("[train]\nloss = cross_entropy\n", file) >= 0);
    CHECK(fclose(file) == 0);

    CHECK(tkm_ini_parse_file(&ini, path) == TKM_OK);
    CHECK(strcmp(tkm_ini_get(&ini, "train", "loss", ""), "cross_entropy") == 0);
    CHECK(tkm_ini_parse_file(&ini, "build/missing_test_ini_parse_file.ini") == TKM_ERR);
    CHECK(tkm_ini_parse_file(0, path) == TKM_ERR);
    CHECK(tkm_ini_parse_file(&ini, 0) == TKM_ERR);
}

static void test_ini_rejects_bad_input_and_bad_typed_values(void) {
    TkmIni ini;
    int32_t i32 = 0;
    float f32 = 0.0f;

    CHECK(tkm_ini_parse(0, "[x]\n") == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, 0) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "kind without equals\n") == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[x]\ni = nope\nf = nope\n") == TKM_OK);
    CHECK(tkm_ini_get_i32(&ini, "x", "i", 0, &i32) == TKM_ERR);
    CHECK(tkm_ini_get_f32(&ini, "x", "f", 0.0f, &f32) == TKM_ERR);
    CHECK(tkm_ini_get_i32(&ini, "x", "i", 0, 0) == TKM_ERR);
    CHECK(tkm_ini_get_f32(&ini, "x", "f", 0.0f, 0) == TKM_ERR);
}

int main(void) {
    test_ini_parses_sections_keys_values_and_comments();
    test_ini_uses_defaults_and_later_entries_override_earlier_entries();
    test_ini_parse_file_loads_project_configs();
    test_ini_rejects_bad_input_and_bad_typed_values();
    puts("ini tests passed");
    return 0;
}
