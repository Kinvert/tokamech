#include "core/tokenizer/int_bins.h"

int tkm_int_bins_init(TkmIntBins* tokenizer, int16_t min_value, int16_t max_value) {
    if (max_value < min_value) {
        return TKM_ERR;
    }

    int32_t size = (int32_t)max_value - (int32_t)min_value + 1;
    if (size <= 0 || size > (int32_t)(TKM_INVALID_TOKEN - 1u)) {
        return TKM_ERR;
    }

    tokenizer->min_value = min_value;
    tokenizer->max_value = max_value;
    tokenizer->vocab_size = (uint16_t)size;
    return TKM_OK;
}

uint16_t tkm_int_bins_encode(const TkmIntBins* tokenizer, int16_t value) {
    if (value < tokenizer->min_value || value > tokenizer->max_value) {
        return TKM_INVALID_TOKEN;
    }

    return (uint16_t)(value - tokenizer->min_value);
}
