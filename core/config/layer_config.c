#include "core/config/layer_config.h"

#include <string.h>

static int tkm_match(const char* text, const char* expected) {
    return text && strcmp(text, expected) == 0;
}

static int tkm_parse_u32(const char* text, uint32_t* out) {
    uint32_t value = 0;

    if (!text || !out || text[0] == '\0') {
        return TKM_ERR;
    }

    for (uint32_t i = 0; text[i] != '\0'; i++) {
        uint32_t digit;

        if (text[i] < '0' || text[i] > '9') {
            return TKM_ERR;
        }
        digit = (uint32_t)(text[i] - '0');
        if (value > (UINT32_MAX - digit) / 10u) {
            return TKM_ERR;
        }
        value = value * 10u + digit;
    }

    *out = value;
    return TKM_OK;
}

int tkm_tokenizer_kind_from_string(const char* text, TkmTokenizerKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "int_bins")) {
        *out = TKM_TOKENIZER_INT_BINS;
        return TKM_OK;
    }
    if (tkm_match(text, "packed_token")) {
        *out = TKM_TOKENIZER_PACKED_TOKEN;
        return TKM_OK;
    }
    if (tkm_match(text, "raw_continuous")) {
        *out = TKM_TOKENIZER_RAW_CONTINUOUS;
        return TKM_OK;
    }
    if (tkm_match(text, "vq_code")) {
        *out = TKM_TOKENIZER_VQ_CODE;
        return TKM_OK;
    }
    if (tkm_match(text, "normalized_continuous")) {
        *out = TKM_TOKENIZER_NORMALIZED_CONTINUOUS;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_embedder_kind_from_string(const char* text, TkmEmbedderKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "none")) {
        *out = TKM_EMBEDDER_NONE;
        return TKM_OK;
    }
    if (tkm_match(text, "lookup")) {
        *out = TKM_EMBEDDER_LOOKUP;
        return TKM_OK;
    }
    if (tkm_match(text, "linear")) {
        *out = TKM_EMBEDDER_LINEAR;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_sequence_layout_kind_from_string(const char* text, TkmSequenceLayoutKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "obs_action_interleaved")) {
        *out = TKM_SEQUENCE_LAYOUT_OBS_ACTION_INTERLEAVED;
        return TKM_OK;
    }
    if (tkm_match(text, "joint_transition")) {
        *out = TKM_SEQUENCE_LAYOUT_JOINT_TRANSITION;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_model_kind_from_string(const char* text, TkmModelKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "lookup_policy")) {
        *out = TKM_MODEL_LOOKUP_POLICY;
        return TKM_OK;
    }
    if (tkm_match(text, "ngram")) {
        *out = TKM_MODEL_NGRAM;
        return TKM_OK;
    }
    if (tkm_match(text, "sparse_lookup")) {
        *out = TKM_MODEL_SPARSE_LOOKUP;
        return TKM_OK;
    }
    if (tkm_match(text, "transformer_decoder")) {
        *out = TKM_MODEL_TRANSFORMER_DECODER;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_head_kind_from_string(const char* text, TkmHeadKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "categorical")) {
        *out = TKM_HEAD_CATEGORICAL;
        return TKM_OK;
    }
    if (tkm_match(text, "multi_categorical")) {
        *out = TKM_HEAD_MULTI_CATEGORICAL;
        return TKM_OK;
    }
    if (tkm_match(text, "scalar_regression")) {
        *out = TKM_HEAD_SCALAR_REGRESSION;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_decoder_kind_from_string(const char* text, TkmDecoderKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "argmax")) {
        *out = TKM_DECODER_ARGMAX;
        return TKM_OK;
    }
    if (tkm_match(text, "sample")) {
        *out = TKM_DECODER_SAMPLE;
        return TKM_OK;
    }
    if (tkm_match(text, "inverse_bins")) {
        *out = TKM_DECODER_INVERSE_BINS;
        return TKM_OK;
    }
    if (tkm_match(text, "discrete_action")) {
        *out = TKM_DECODER_DISCRETE_ACTION;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_loss_kind_from_string(const char* text, TkmConfiguredLossKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_match(text, "cross_entropy")) {
        *out = TKM_LOSS_CROSS_ENTROPY;
        return TKM_OK;
    }
    if (tkm_match(text, "mse")) {
        *out = TKM_LOSS_MSE;
        return TKM_OK;
    }
    if (tkm_match(text, "huber")) {
        *out = TKM_LOSS_HUBER;
        return TKM_OK;
    }
    if (tkm_match(text, "binary_cross_entropy")) {
        *out = TKM_LOSS_BINARY_CROSS_ENTROPY;
        return TKM_OK;
    }
    if (tkm_match(text, "weighted_sum")) {
        *out = TKM_LOSS_WEIGHTED_SUM;
        return TKM_OK;
    }
    if (tkm_match(text, "action_smoothness")) {
        *out = TKM_LOSS_ACTION_SMOOTHNESS;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_layer_config_from_ini(const TkmIni* ini, TkmLayerConfig* out) {
    const char* tokenizer;
    const char* embedder;
    const char* sequence_layout;
    const char* model;
    const char* model_hidden_dim;
    const char* head;
    const char* decoder;
    const char* loss;

    if (!ini || !out) {
        return TKM_ERR;
    }

    tokenizer = tkm_ini_get(ini, "tokenizer", "kind", "int_bins");
    embedder = tkm_ini_get(ini, "embedder", "kind", "none");
    sequence_layout = tkm_ini_get(ini, "sequence_layout", "kind", "obs_action_interleaved");
    model = tkm_ini_get(ini, "model", "kind", "lookup_policy");
    model_hidden_dim = tkm_ini_get(ini, "model", "hidden_dim", "32");
    head = tkm_ini_get(ini, "head", "kind", "categorical");
    decoder = tkm_ini_get(ini, "decoder", "kind", "argmax");
    loss = tkm_ini_get(ini, "loss", "kind", "cross_entropy");

    if (tkm_tokenizer_kind_from_string(tokenizer, &out->tokenizer) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_embedder_kind_from_string(embedder, &out->embedder) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_sequence_layout_kind_from_string(sequence_layout, &out->sequence_layout) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_model_kind_from_string(model, &out->model) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_parse_u32(model_hidden_dim, &out->model_hidden_dim) != TKM_OK ||
        out->model_hidden_dim == 0 ||
        out->model_hidden_dim > TKM_LAYER_MAX_MODEL_HIDDEN_DIM) {
        return TKM_ERR;
    }
    if (tkm_head_kind_from_string(head, &out->head) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_decoder_kind_from_string(decoder, &out->decoder) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_loss_kind_from_string(loss, &out->loss) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}
