#include "core/config/train_config.h"

#include <string.h>

static int tkm_train_config_match(const char* text, const char* expected) {
    return text && strcmp(text, expected) == 0;
}

static int tkm_train_config_parse_u32(const char* text, uint32_t* out) {
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

static int tkm_train_config_parse_positive_f32(const char* text, float* out) {
    float value = 0.0f;
    float scale = 0.1f;
    uint8_t seen_digit = 0;
    uint8_t seen_dot = 0;

    if (!text || !out || text[0] == '\0') {
        return TKM_ERR;
    }

    for (uint32_t i = 0; text[i] != '\0'; i++) {
        if (text[i] == '.') {
            if (seen_dot) {
                return TKM_ERR;
            }
            seen_dot = 1;
            continue;
        }
        if (text[i] < '0' || text[i] > '9') {
            return TKM_ERR;
        }
        seen_digit = 1;
        if (seen_dot) {
            value += (float)(text[i] - '0') * scale;
            scale *= 0.1f;
        } else {
            value = value * 10.0f + (float)(text[i] - '0');
        }
    }

    if (!seen_digit || value <= 0.0f || value > TKM_TRAIN_MAX_LEARNING_RATE) {
        return TKM_ERR;
    }

    *out = value;
    return TKM_OK;
}

static int tkm_train_config_parse_learning_rate(const char* text, float* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_train_config_match(text, "auto")) {
        *out = TKM_TRAIN_AUTO_LEARNING_RATE;
        return TKM_OK;
    }
    return tkm_train_config_parse_positive_f32(text, out);
}

int tkm_train_loss_kind_from_string(const char* text, TkmTrainLossKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_train_config_match(text, "cross_entropy")) {
        *out = TKM_TRAIN_LOSS_CROSS_ENTROPY;
        return TKM_OK;
    }
    if (tkm_train_config_match(text, "masked_cross_entropy")) {
        *out = TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY;
        return TKM_OK;
    }
    if (tkm_train_config_match(text, "mse")) {
        *out = TKM_TRAIN_LOSS_MSE;
        return TKM_OK;
    }
    if (tkm_train_config_match(text, "margin")) {
        *out = TKM_TRAIN_LOSS_MARGIN;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_train_config_from_ini(const TkmIni* ini, TkmTrainConfig* out) {
    const char* loss;
    const char* rollout_steps;
    const char* dagger_rounds;
    const char* learning_rate;
    const char* dagger_learning_rate;

    if (!ini || !out) {
        return TKM_ERR;
    }

    loss = tkm_ini_get(ini, "train", "loss", "cross_entropy");
    if (tkm_train_loss_kind_from_string(loss, &out->loss) != TKM_OK) {
        return TKM_ERR;
    }

    rollout_steps = tkm_ini_get(ini, "train", "rollout_steps", "96");
    if (tkm_train_config_parse_u32(rollout_steps, &out->rollout_steps) != TKM_OK ||
        out->rollout_steps == 0 ||
        out->rollout_steps > TKM_TRAIN_MAX_ROLLOUT_STEPS) {
        return TKM_ERR;
    }

    dagger_rounds = tkm_ini_get(ini, "train", "dagger_rounds", "4");
    if (tkm_train_config_parse_u32(dagger_rounds, &out->dagger_rounds) != TKM_OK ||
        out->dagger_rounds == 0 ||
        out->dagger_rounds > TKM_TRAIN_MAX_DAGGER_ROUNDS) {
        return TKM_ERR;
    }

    learning_rate = tkm_ini_get(ini, "train", "learning_rate", "auto");
    if (tkm_train_config_parse_learning_rate(learning_rate, &out->learning_rate) != TKM_OK) {
        return TKM_ERR;
    }

    dagger_learning_rate = tkm_ini_get(ini, "train", "dagger_learning_rate", "0.005");
    if (tkm_train_config_parse_positive_f32(dagger_learning_rate, &out->dagger_learning_rate) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}
