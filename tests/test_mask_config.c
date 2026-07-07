#include <stdio.h>
#include <stdlib.h>

#include "core/config/mask_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_mask_config_reads_strategy_from_ini(void) {
    const char* text =
        "[mask]\n"
        "strategy = survival\n";
    TkmIni ini;
    TkmMaskConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_mask_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.strategy == TKM_MASK_STRATEGY_SURVIVAL);
}

static void test_mask_config_uses_immediate_default(void) {
    TkmIni ini;
    TkmMaskConfig config;

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_mask_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.strategy == TKM_MASK_STRATEGY_IMMEDIATE);
}

static void test_mask_strategy_parser_accepts_interchangeable_options(void) {
    TkmMaskStrategyKind strategy;
    TkmIni ini;
    TkmMaskConfig config;

    CHECK(tkm_mask_strategy_kind_from_string("immediate", &strategy) == TKM_OK);
    CHECK(strategy == TKM_MASK_STRATEGY_IMMEDIATE);
    CHECK(tkm_mask_strategy_kind_from_string("survival", &strategy) == TKM_OK);
    CHECK(strategy == TKM_MASK_STRATEGY_SURVIVAL);

    CHECK(tkm_mask_strategy_kind_from_string(0, &strategy) == TKM_ERR);
    CHECK(tkm_mask_strategy_kind_from_string("unknown", &strategy) == TKM_ERR);
    CHECK(tkm_mask_strategy_kind_from_string("survival", 0) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[mask]\nstrategy = unknown\n") == TKM_OK);
    CHECK(tkm_mask_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_mask_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_mask_config_from_ini(&ini, 0) == TKM_ERR);
}

int main(void) {
    test_mask_config_reads_strategy_from_ini();
    test_mask_config_uses_immediate_default();
    test_mask_strategy_parser_accepts_interchangeable_options();
    puts("mask config tests passed");
    return 0;
}
