#include "core/model/ngram.h"

static uint32_t tkm_ngram_offset(const TkmNgramModel* model, uint32_t prev_token, uint32_t next_token) {
    return prev_token * model->vocab_size + next_token;
}

static int tkm_ngram_ready(const TkmNgramModel* model) {
    return model && model->vocab_size > 0 && model->vocab_size <= TKM_NGRAM_MAX_VOCAB;
}

int tkm_ngram_init(TkmNgramModel* model, uint32_t vocab_size) {
    if (!model || vocab_size == 0 || vocab_size > TKM_NGRAM_MAX_VOCAB) {
        return TKM_ERR;
    }

    model->vocab_size = vocab_size;
    for (uint32_t i = 0; i < TKM_NGRAM_MAX_VOCAB * TKM_NGRAM_MAX_VOCAB; i++) {
        model->counts[i] = 0;
    }

    return TKM_OK;
}

int tkm_ngram_train_sequence(TkmNgramModel* model, const uint16_t* tokens, uint32_t token_count) {
    if (!tkm_ngram_ready(model) || !tokens || token_count < 2) {
        return TKM_ERR;
    }

    for (uint32_t i = 1; i < token_count; i++) {
        uint16_t prev = tokens[i - 1];
        uint16_t next = tokens[i];
        if (prev >= model->vocab_size || next >= model->vocab_size) {
            return TKM_ERR;
        }
        model->counts[tkm_ngram_offset(model, prev, next)]++;
    }

    return TKM_OK;
}

int tkm_ngram_predict_next(const TkmNgramModel* model, uint16_t prev_token, uint16_t* out_token) {
    uint16_t best_token = 0;
    uint32_t best_count = 0;

    if (!tkm_ngram_ready(model) || !out_token || prev_token >= model->vocab_size) {
        return TKM_ERR;
    }

    for (uint32_t next = 0; next < model->vocab_size; next++) {
        uint32_t count = model->counts[tkm_ngram_offset(model, prev_token, next)];
        if (count > best_count) {
            best_count = count;
            best_token = (uint16_t)next;
        }
    }

    *out_token = best_token;
    return TKM_OK;
}
