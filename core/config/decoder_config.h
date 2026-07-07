#ifndef TKM_DECODER_CONFIG_H
#define TKM_DECODER_CONFIG_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/config/ini.h"

#define TKM_DECODER_DEFAULT_ROLLOUT_HORIZON 16u
#define TKM_DECODER_MAX_ROLLOUT_HORIZON 512u

typedef enum {
    TKM_DECODER_FALLBACK_NONE = 0,
    TKM_DECODER_FALLBACK_PLANNER_SCORE = 1,
    TKM_DECODER_FALLBACK_BEST_SCORE = 2,
    TKM_DECODER_FALLBACK_ROLLOUT_SCORE = 3,
} TkmDecoderFallbackKind;

typedef enum {
    TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS = 0,
    TKM_DECODER_ROLLOUT_SCORE_SPACE_DISTANCE = 1,
} TkmDecoderRolloutScoreKind;

typedef struct {
    TkmDecoderFallbackKind fallback;
    uint32_t rollout_horizon;
    TkmDecoderRolloutScoreKind rollout_score_mode;
} TkmDecoderConfig;

int tkm_decoder_fallback_kind_from_string(const char* text, TkmDecoderFallbackKind* out);
int tkm_decoder_rollout_score_kind_from_string(const char* text, TkmDecoderRolloutScoreKind* out);
int tkm_decoder_config_from_ini(const TkmIni* ini, TkmDecoderConfig* out);

#endif
