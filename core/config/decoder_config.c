#include "core/config/decoder_config.h"

#include <string.h>

static int tkm_decoder_config_match(const char* text, const char* expected) {
    return text && strcmp(text, expected) == 0;
}

static int tkm_decoder_config_parse_u32(const char* text, uint32_t* out) {
    uint32_t value = 0;

    if (!text || !out || text[0] == '\0') {
        return TKM_ERR;
    }

    for (uint32_t i = 0; text[i] != '\0'; i++) {
        uint32_t digit;

        if (text[i] < '0' || text[i] > '9') {
            return TKM_ERR;
        }
        digit = (uint32_t)(text[i] - '0');
        if (value > (UINT32_MAX - digit) / 10u) {
            return TKM_ERR;
        }
        value = value * 10u + digit;
    }

    *out = value;
    return TKM_OK;
}

int tkm_decoder_fallback_kind_from_string(const char* text, TkmDecoderFallbackKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_decoder_config_match(text, "none")) {
        *out = TKM_DECODER_FALLBACK_NONE;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_decoder_config_from_ini(const TkmIni* ini, TkmDecoderConfig* out) {
    const char* fallback;
    const char* rollout_horizon;

    if (!ini || !out) {
        return TKM_ERR;
    }

    fallback = tkm_ini_get(ini, "decoder", "fallback", "none");
    if (tkm_decoder_fallback_kind_from_string(fallback, &out->fallback) != TKM_OK) {
        return TKM_ERR;
    }

    rollout_horizon = tkm_ini_get(ini, "decoder", "rollout_horizon", "16");
    if (tkm_decoder_config_parse_u32(rollout_horizon, &out->rollout_horizon) != TKM_OK ||
        out->rollout_horizon == 0u ||
        out->rollout_horizon > TKM_DECODER_MAX_ROLLOUT_HORIZON) {
        return TKM_ERR;
    }

    return TKM_OK;
}
