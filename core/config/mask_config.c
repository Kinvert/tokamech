#include "core/config/mask_config.h"

#include <string.h>

static int tkm_mask_config_match(const char* text, const char* expected) {
    return text && strcmp(text, expected) == 0;
}

int tkm_mask_strategy_kind_from_string(const char* text, TkmMaskStrategyKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_mask_config_match(text, "immediate")) {
        *out = TKM_MASK_STRATEGY_IMMEDIATE;
        return TKM_OK;
    }
    if (tkm_mask_config_match(text, "survival")) {
        *out = TKM_MASK_STRATEGY_SURVIVAL;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_mask_config_from_ini(const TkmIni* ini, TkmMaskConfig* out) {
    const char* strategy;

    if (!ini || !out) {
        return TKM_ERR;
    }

    strategy = tkm_ini_get(ini, "mask", "strategy", "immediate");
    if (tkm_mask_strategy_kind_from_string(strategy, &out->strategy) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}
