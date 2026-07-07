#include "core/embedder/embedder.h"

static uint32_t tkm_lookup_offset(const TkmLookupEmbedder* embedder, uint32_t token, uint32_t dim_index) {
    return token * embedder->dim + dim_index;
}

static uint32_t tkm_linear_offset(const TkmLinearEmbedder* embedder, uint32_t input_index, uint32_t output_index) {
    return input_index * embedder->output_dim + output_index;
}

static int tkm_lookup_ready(const TkmLookupEmbedder* embedder) {
    return embedder && embedder->token_count > 0 && embedder->dim > 0;
}

static int tkm_linear_ready(const TkmLinearEmbedder* embedder) {
    return embedder && embedder->input_dim > 0 && embedder->output_dim > 0;
}

int tkm_lookup_embedder_init(TkmLookupEmbedder* embedder, uint32_t token_count, uint32_t dim, const float* table) {
    if (!embedder || !table || token_count == 0 || dim == 0) {
        return TKM_ERR;
    }
    if (token_count > TKM_EMBED_MAX_TOKENS || dim > TKM_EMBED_MAX_DIM) {
        return TKM_ERR;
    }

    embedder->token_count = token_count;
    embedder->dim = dim;

    for (uint32_t token = 0; token < token_count; token++) {
        for (uint32_t d = 0; d < dim; d++) {
            embedder->table[tkm_lookup_offset(embedder, token, d)] = table[token * dim + d];
        }
    }

    return TKM_OK;
}

int tkm_lookup_embedder_encode(const TkmLookupEmbedder* embedder, uint16_t token, float* out, uint32_t out_count) {
    if (!tkm_lookup_ready(embedder) || !out || token >= embedder->token_count || out_count < embedder->dim) {
        return TKM_ERR;
    }

    for (uint32_t d = 0; d < embedder->dim; d++) {
        out[d] = embedder->table[tkm_lookup_offset(embedder, token, d)];
    }

    return TKM_OK;
}

int tkm_lookup_embedder_encode_batch(
    const TkmLookupEmbedder* embedder,
    const uint16_t* tokens,
    uint32_t token_count,
    float* out,
    uint32_t out_count
) {
    if (!tkm_lookup_ready(embedder) || !tokens || !out || token_count == 0 || out_count < token_count * embedder->dim) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < token_count; i++) {
        if (tkm_lookup_embedder_encode(embedder, tokens[i], out + i * embedder->dim, embedder->dim) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int tkm_linear_embedder_init(
    TkmLinearEmbedder* embedder,
    uint32_t input_dim,
    uint32_t output_dim,
    const float* weights,
    const float* bias
) {
    if (!embedder || !weights || input_dim == 0 || output_dim == 0) {
        return TKM_ERR;
    }
    if (input_dim > TKM_LINEAR_MAX_INPUT || output_dim > TKM_EMBED_MAX_DIM) {
        return TKM_ERR;
    }

    embedder->input_dim = input_dim;
    embedder->output_dim = output_dim;

    for (uint32_t i = 0; i < input_dim; i++) {
        for (uint32_t o = 0; o < output_dim; o++) {
            embedder->weights[tkm_linear_offset(embedder, i, o)] = weights[i * output_dim + o];
        }
    }

    for (uint32_t o = 0; o < output_dim; o++) {
        embedder->bias[o] = bias ? bias[o] : 0.0f;
    }

    return TKM_OK;
}

int tkm_linear_embedder_encode(const TkmLinearEmbedder* embedder, const float* input, float* out, uint32_t out_count) {
    if (!tkm_linear_ready(embedder) || !input || !out || out_count < embedder->output_dim) {
        return TKM_ERR;
    }

    for (uint32_t o = 0; o < embedder->output_dim; o++) {
        float value = embedder->bias[o];
        for (uint32_t i = 0; i < embedder->input_dim; i++) {
            value += input[i] * embedder->weights[tkm_linear_offset(embedder, i, o)];
        }
        out[o] = value;
    }

    return TKM_OK;
}
