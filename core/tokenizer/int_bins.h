#ifndef TKM_INT_BINS_H
#define TKM_INT_BINS_H

#include <stdint.h>

#include "core/common/status.h"

typedef struct {
    int16_t min_value;
    int16_t max_value;
    uint16_t vocab_size;
} TkmIntBins;

int tkm_int_bins_init(TkmIntBins* tokenizer, int16_t min_value, int16_t max_value);
uint16_t tkm_int_bins_encode(const TkmIntBins* tokenizer, int16_t value);

#endif
