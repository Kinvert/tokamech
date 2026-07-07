#include "core/vectorizer/continuous.h"

static int tkm_continuous_dim_valid(uint32_t dim) {
    return dim > 0 && dim <= TKM_CONTINUOUS_MAX_DIM;
}

static int tkm_continuous_ready(const TkmContinuousVectorizer* vectorizer) {
    return vectorizer && tkm_continuous_dim_valid(vectorizer->dim);
}

int tkm_continuous_vectorizer_init_raw(TkmContinuousVectorizer* vectorizer, uint32_t dim) {
    if (!vectorizer || !tkm_continuous_dim_valid(dim)) {
        return TKM_ERR;
    }

    vectorizer->kind = TKM_CONTINUOUS_RAW;
    vectorizer->dim = dim;

    for (uint32_t i = 0; i < dim; i++) {
        vectorizer->mean[i] = 0.0f;
        vectorizer->inv_std[i] = 1.0f;
    }

    return TKM_OK;
}

int tkm_continuous_vectorizer_init_standardize(
    TkmContinuousVectorizer* vectorizer,
    uint32_t dim,
    const float* mean,
    const float* std
) {
    if (!vectorizer || !tkm_continuous_dim_valid(dim) || !mean || !std) {
        return TKM_ERR;
    }

    vectorizer->kind = TKM_CONTINUOUS_STANDARDIZE;
    vectorizer->dim = dim;

    for (uint32_t i = 0; i < dim; i++) {
        if (std[i] <= 0.0f) {
            return TKM_ERR;
        }
        vectorizer->mean[i] = mean[i];
        vectorizer->inv_std[i] = 1.0f / std[i];
    }

    return TKM_OK;
}

int tkm_continuous_vectorizer_apply(
    const TkmContinuousVectorizer* vectorizer,
    const float* input,
    float* output,
    uint32_t output_count
) {
    if (!tkm_continuous_ready(vectorizer) || !input || !output || output_count < vectorizer->dim) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < vectorizer->dim; i++) {
        output[i] = (input[i] - vectorizer->mean[i]) * vectorizer->inv_std[i];
    }

    return TKM_OK;
}
