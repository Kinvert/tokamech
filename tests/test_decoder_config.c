#include <stdio.h>
#include <stdlib.h>

#include "core/config/decoder_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_decoder_config_reads_none_fallback_from_ini(void) {
    const char* text =
        "[decoder]\n"
        "fallback = none\n"
        "rollout_horizon = 24\n";
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(config.rollout_horizon == 24);
}

static void test_decoder_config_uses_none_default(void) {
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(config.rollout_horizon == 16);
}

static void test_decoder_fallback_parser_rejects_runtime_shortcuts(void) {
    TkmDecoderFallbackKind fallback;
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_decoder_fallback_kind_from_string("none", &fallback) == TKM_OK);
    CHECK(fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(tkm_decoder_fallback_kind_from_string("planner_score", &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("best_score", &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("rollout_score", &fallback) == TKM_ERR);

    CHECK(tkm_decoder_fallback_kind_from_string(0, &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("unknown", &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("none", 0) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[decoder]\nfallback = unknown\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[decoder]\nrollout_horizon = 0\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[decoder]\nrollout_horizon = many\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_decoder_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_decoder_config_from_ini(&ini, 0) == TKM_ERR);
}

int main(void) {
    test_decoder_config_reads_none_fallback_from_ini();
    test_decoder_config_uses_none_default();
    test_decoder_fallback_parser_rejects_runtime_shortcuts();
    puts("decoder config tests passed");
    return 0;
}
