#ifndef TKM_LAYER_CONFIG_H
#define TKM_LAYER_CONFIG_H

#include <stdint.h>

#include "core/config/ini.h"
#include "core/common/status.h"

#define TKM_LAYER_DEFAULT_MODEL_HIDDEN_DIM 32u
#define TKM_LAYER_MAX_MODEL_HIDDEN_DIM 256u

typedef enum {
    TKM_TOKENIZER_INT_BINS = 0,
    TKM_TOKENIZER_PACKED_TOKEN = 1,
    TKM_TOKENIZER_RAW_CONTINUOUS = 2,
    TKM_TOKENIZER_VQ_CODE = 3,
    TKM_TOKENIZER_NORMALIZED_CONTINUOUS = 4,
} TkmTokenizerKind;

typedef enum {
    TKM_EMBEDDER_NONE = 0,
    TKM_EMBEDDER_LOOKUP = 1,
    TKM_EMBEDDER_LINEAR = 2,
} TkmEmbedderKind;

typedef enum {
    TKM_SEQUENCE_LAYOUT_OBS_ACTION_INTERLEAVED = 0,
    TKM_SEQUENCE_LAYOUT_JOINT_TRANSITION = 1,
} TkmSequenceLayoutKind;

typedef enum {
    TKM_MODEL_LOOKUP_POLICY = 0,
    TKM_MODEL_NGRAM = 2,
    TKM_MODEL_SPARSE_LOOKUP = 4,
    TKM_MODEL_TRANSFORMER_DECODER = 8,
} TkmModelKind;

typedef enum {
    TKM_HEAD_CATEGORICAL = 0,
    TKM_HEAD_MULTI_CATEGORICAL = 1,
    TKM_HEAD_SCALAR_REGRESSION = 2,
} TkmHeadKind;

typedef enum {
    TKM_DECODER_ARGMAX = 0,
    TKM_DECODER_SAMPLE = 1,
    TKM_DECODER_INVERSE_BINS = 2,
    TKM_DECODER_DISCRETE_ACTION = 3,
} TkmDecoderKind;

typedef enum {
    TKM_LOSS_CROSS_ENTROPY = 0,
    TKM_LOSS_MSE = 1,
    TKM_LOSS_HUBER = 2,
    TKM_LOSS_BINARY_CROSS_ENTROPY = 3,
    TKM_LOSS_WEIGHTED_SUM = 4,
    TKM_LOSS_ACTION_SMOOTHNESS = 5,
} TkmConfiguredLossKind;

typedef struct {
    TkmTokenizerKind tokenizer;
    TkmEmbedderKind embedder;
    TkmSequenceLayoutKind sequence_layout;
    TkmModelKind model;
    uint32_t model_hidden_dim;
    TkmHeadKind head;
    TkmDecoderKind decoder;
    TkmConfiguredLossKind loss;
} TkmLayerConfig;

int tkm_tokenizer_kind_from_string(const char* text, TkmTokenizerKind* out);
int tkm_embedder_kind_from_string(const char* text, TkmEmbedderKind* out);
int tkm_sequence_layout_kind_from_string(const char* text, TkmSequenceLayoutKind* out);
int tkm_model_kind_from_string(const char* text, TkmModelKind* out);
int tkm_head_kind_from_string(const char* text, TkmHeadKind* out);
int tkm_decoder_kind_from_string(const char* text, TkmDecoderKind* out);
int tkm_loss_kind_from_string(const char* text, TkmConfiguredLossKind* out);
int tkm_layer_config_from_ini(const TkmIni* ini, TkmLayerConfig* out);

#endif
