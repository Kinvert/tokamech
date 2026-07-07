#ifndef TKM_CONTINUOUS_VECTORIZER_H
#define TKM_CONTINUOUS_VECTORIZER_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_CONTINUOUS_MAX_DIM 256

#define TKM_CONTINUOUS_RAW 0
#define TKM_CONTINUOUS_STANDARDIZE 1

typedef struct {
    uint8_t kind;
    uint32_t dim;
    float mean[TKM_CONTINUOUS_MAX_DIM];
    float inv_std[TKM_CONTINUOUS_MAX_DIM];
} TkmContinuousVectorizer;

int tkm_continuous_vectorizer_init_raw(TkmContinuousVectorizer* vectorizer, uint32_t dim);
int tkm_continuous_vectorizer_init_standardize(
    TkmContinuousVectorizer* vectorizer,
    uint32_t dim,
    const float* mean,
    const float* std
);
int tkm_continuous_vectorizer_apply(
    const TkmContinuousVectorizer* vectorizer,
    const float* input,
    float* output,
    uint32_t output_count
);

#endif
