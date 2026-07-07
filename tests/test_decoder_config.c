#include <stdio.h>
#include <stdlib.h>

#include "core/config/decoder_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_decoder_config_reads_fallback_from_ini(void) {
    const char* text =
        "[decoder]\n"
        "fallback = planner_score\n"
        "rollout_horizon = 24\n"
        "rollout_score_mode = space_distance\n";
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.fallback == TKM_DECODER_FALLBACK_PLANNER_SCORE);
    CHECK(config.rollout_horizon == 24);
    CHECK(config.rollout_score_mode == TKM_DECODER_ROLLOUT_SCORE_SPACE_DISTANCE);
}

static void test_decoder_config_uses_none_default(void) {
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(config.rollout_horizon == 16);
    CHECK(config.rollout_score_mode == TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS);
}

static void test_decoder_fallback_parser_accepts_interchangeable_options(void) {
    TkmDecoderFallbackKind fallback;
    TkmIni ini;
    TkmDecoderConfig config;

    CHECK(tkm_decoder_fallback_kind_from_string("none", &fallback) == TKM_OK);
    CHECK(fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(tkm_decoder_fallback_kind_from_string("planner_score", &fallback) == TKM_OK);
    CHECK(fallback == TKM_DECODER_FALLBACK_PLANNER_SCORE);
    CHECK(tkm_decoder_fallback_kind_from_string("best_score", &fallback) == TKM_OK);
    CHECK(fallback == TKM_DECODER_FALLBACK_BEST_SCORE);
    CHECK(tkm_decoder_fallback_kind_from_string("rollout_score", &fallback) == TKM_OK);
    CHECK(fallback == TKM_DECODER_FALLBACK_ROLLOUT_SCORE);
    TkmDecoderRolloutScoreKind rollout_score_mode;
    CHECK(tkm_decoder_rollout_score_kind_from_string("food_steps", &rollout_score_mode) == TKM_OK);
    CHECK(rollout_score_mode == TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS);
    CHECK(tkm_decoder_rollout_score_kind_from_string("space_distance", &rollout_score_mode) == TKM_OK);
    CHECK(rollout_score_mode == TKM_DECODER_ROLLOUT_SCORE_SPACE_DISTANCE);

    CHECK(tkm_decoder_fallback_kind_from_string(0, &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("unknown", &fallback) == TKM_ERR);
    CHECK(tkm_decoder_fallback_kind_from_string("none", 0) == TKM_ERR);
    CHECK(tkm_decoder_rollout_score_kind_from_string(0, &rollout_score_mode) == TKM_ERR);
    CHECK(tkm_decoder_rollout_score_kind_from_string("unknown", &rollout_score_mode) == TKM_ERR);
    CHECK(tkm_decoder_rollout_score_kind_from_string("food_steps", 0) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[decoder]\nfallback = unknown\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[decoder]\nrollout_horizon = 0\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[decoder]\nrollout_horizon = many\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[decoder]\nrollout_score_mode = unknown\n") == TKM_OK);
    CHECK(tkm_decoder_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_decoder_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_decoder_config_from_ini(&ini, 0) == TKM_ERR);
}

int main(void) {
    test_decoder_config_reads_fallback_from_ini();
    test_decoder_config_uses_none_default();
    test_decoder_fallback_parser_accepts_interchangeable_options();
    puts("decoder config tests passed");
    return 0;
}
