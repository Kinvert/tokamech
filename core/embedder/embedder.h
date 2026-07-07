#ifndef TKM_EMBEDDER_H
#define TKM_EMBEDDER_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_EMBED_MAX_TOKENS 1024
#define TKM_EMBED_MAX_DIM 64
#define TKM_LINEAR_MAX_INPUT 128

typedef struct {
    uint32_t token_count;
    uint32_t dim;
    float table[TKM_EMBED_MAX_TOKENS * TKM_EMBED_MAX_DIM];
} TkmLookupEmbedder;

typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    float weights[TKM_LINEAR_MAX_INPUT * TKM_EMBED_MAX_DIM];
    float bias[TKM_EMBED_MAX_DIM];
} TkmLinearEmbedder;

int tkm_lookup_embedder_init(TkmLookupEmbedder* embedder, uint32_t token_count, uint32_t dim, const float* table);
int tkm_lookup_embedder_encode(const TkmLookupEmbedder* embedder, uint16_t token, float* out, uint32_t out_count);
int tkm_lookup_embedder_encode_batch(
    const TkmLookupEmbedder* embedder,
    const uint16_t* tokens,
    uint32_t token_count,
    float* out,
    uint32_t out_count
);

int tkm_linear_embedder_init(
    TkmLinearEmbedder* embedder,
    uint32_t input_dim,
    uint32_t output_dim,
    const float* weights,
    const float* bias
);
int tkm_linear_embedder_encode(const TkmLinearEmbedder* embedder, const float* input, float* out, uint32_t out_count);

#endif
