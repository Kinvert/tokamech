#ifndef TKM_NGRAM_H
#define TKM_NGRAM_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_NGRAM_MAX_VOCAB 256

typedef struct {
    uint32_t vocab_size;
    uint32_t counts[TKM_NGRAM_MAX_VOCAB * TKM_NGRAM_MAX_VOCAB];
} TkmNgramModel;

int tkm_ngram_init(TkmNgramModel* model, uint32_t vocab_size);
int tkm_ngram_train_sequence(TkmNgramModel* model, const uint16_t* tokens, uint32_t token_count);
int tkm_ngram_predict_next(const TkmNgramModel* model, uint16_t prev_token, uint16_t* out_token);

#endif
