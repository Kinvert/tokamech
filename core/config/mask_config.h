#ifndef TKM_MASK_CONFIG_H
#define TKM_MASK_CONFIG_H

#include "core/common/status.h"
#include "core/config/ini.h"

typedef enum {
    TKM_MASK_STRATEGY_NONE = 0,
} TkmMaskStrategyKind;

typedef struct {
    TkmMaskStrategyKind strategy;
} TkmMaskConfig;

int tkm_mask_strategy_kind_from_string(const char* text, TkmMaskStrategyKind* out);
int tkm_mask_config_from_ini(const TkmIni* ini, TkmMaskConfig* out);

#endif
