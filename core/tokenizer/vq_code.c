#include "core/tokenizer/vq_code.h"

static uint32_t tkm_vq_offset(const TkmVqCode* vq, uint32_t code, uint32_t dim_index) {
    return code * vq->dim + dim_index;
}

static int tkm_vq_is_ready(const TkmVqCode* vq) {
    return vq && vq->code_count > 0 && vq->dim > 0;
}

int tkm_vq_code_init(TkmVqCode* vq, uint32_t code_count, uint32_t dim, const float* codebook) {
    if (!vq || !codebook || code_count == 0 || dim == 0) {
        return TKM_ERR;
    }
    if (code_count > TKM_VQ_MAX_CODES || dim > TKM_VQ_MAX_DIM) {
        return TKM_ERR;
    }

    vq->code_count = code_count;
    vq->dim = dim;

    for (uint32_t code = 0; code < code_count; code++) {
        for (uint32_t d = 0; d < dim; d++) {
            vq->codebook[tkm_vq_offset(vq, code, d)] = codebook[code * dim + d];
        }
    }

    return TKM_OK;
}

int tkm_vq_code_encode(const TkmVqCode* vq, const float* vector, uint16_t* out_code, float* out_distance) {
    uint16_t best_code = 0;
    float best_distance = 0.0f;

    if (!tkm_vq_is_ready(vq) || !vector || !out_code) {
        return TKM_ERR;
    }

    for (uint32_t code = 0; code < vq->code_count; code++) {
        float distance = 0.0f;

        for (uint32_t d = 0; d < vq->dim; d++) {
            float diff = vector[d] - vq->codebook[tkm_vq_offset(vq, code, d)];
            distance += diff * diff;
        }

        if (code == 0 || distance < best_distance) {
            best_code = (uint16_t)code;
            best_distance = distance;
        }
    }

    *out_code = best_code;
    if (out_distance) {
        *out_distance = best_distance;
    }
    return TKM_OK;
}

int tkm_vq_code_decode(const TkmVqCode* vq, uint16_t code, float* out_vector, uint32_t out_count) {
    if (!tkm_vq_is_ready(vq) || !out_vector || code >= vq->code_count || out_count < vq->dim) {
        return TKM_ERR;
    }

    for (uint32_t d = 0; d < vq->dim; d++) {
        out_vector[d] = vq->codebook[tkm_vq_offset(vq, code, d)];
    }

    return TKM_OK;
}

int tkm_vq_code_encode_batch(const TkmVqCode* vq, const float* vectors, uint32_t vector_count, uint16_t* out_codes) {
    if (!tkm_vq_is_ready(vq) || !vectors || !out_codes || vector_count == 0) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < vector_count; i++) {
        const float* vector = vectors + i * vq->dim;
        if (tkm_vq_code_encode(vq, vector, &out_codes[i], 0) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}
