#ifndef TKM_VQ_CODE_H
#define TKM_VQ_CODE_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_VQ_MAX_CODES 256
#define TKM_VQ_MAX_DIM 32

typedef struct {
    uint32_t code_count;
    uint32_t dim;
    float codebook[TKM_VQ_MAX_CODES * TKM_VQ_MAX_DIM];
} TkmVqCode;

int tkm_vq_code_init(TkmVqCode* vq, uint32_t code_count, uint32_t dim, const float* codebook);
int tkm_vq_code_encode(const TkmVqCode* vq, const float* vector, uint16_t* out_code, float* out_distance);
int tkm_vq_code_decode(const TkmVqCode* vq, uint16_t code, float* out_vector, uint32_t out_count);
int tkm_vq_code_encode_batch(const TkmVqCode* vq, const float* vectors, uint32_t vector_count, uint16_t* out_codes);

#endif
