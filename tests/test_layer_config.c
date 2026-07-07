#include <stdio.h>
#include <stdlib.h>

#include "core/config/layer_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_layer_config_reads_known_kinds_from_ini(void) {
    const char* text =
        "[tokenizer]\n"
        "kind = vq_code\n"
        "[embedder]\n"
        "kind = linear\n"
        "[sequence_layout]\n"
        "kind = joint_transition\n"
        "[model]\n"
        "kind = planner_oracle\n"
        "hidden_dim = 64\n"
        "action_features = food_space\n"
        "[head]\n"
        "kind = categorical\n"
        "[decoder]\n"
        "kind = argmax\n"
        "[loss]\n"
        "kind = huber\n";
    TkmIni ini;
    TkmLayerConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.tokenizer == TKM_TOKENIZER_VQ_CODE);
    CHECK(config.embedder == TKM_EMBEDDER_LINEAR);
    CHECK(config.sequence_layout == TKM_SEQUENCE_LAYOUT_JOINT_TRANSITION);
    CHECK(config.model == TKM_MODEL_PLANNER_ORACLE);
    CHECK(config.model_hidden_dim == 64);
    CHECK(config.action_features == TKM_ACTION_FEATURE_FOOD_SPACE);
    CHECK(config.head == TKM_HEAD_CATEGORICAL);
    CHECK(config.decoder == TKM_DECODER_ARGMAX);
    CHECK(config.loss == TKM_LOSS_HUBER);
}

static void test_layer_config_uses_phase0_defaults(void) {
    TkmIni ini;
    TkmLayerConfig config;

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.tokenizer == TKM_TOKENIZER_INT_BINS);
    CHECK(config.embedder == TKM_EMBEDDER_NONE);
    CHECK(config.sequence_layout == TKM_SEQUENCE_LAYOUT_OBS_ACTION_INTERLEAVED);
    CHECK(config.model == TKM_MODEL_LOOKUP_POLICY);
    CHECK(config.model_hidden_dim == 32);
    CHECK(config.action_features == TKM_ACTION_FEATURE_SPACE);
    CHECK(config.head == TKM_HEAD_CATEGORICAL);
    CHECK(config.decoder == TKM_DECODER_ARGMAX);
    CHECK(config.loss == TKM_LOSS_CROSS_ENTROPY);
}

static void test_layer_kind_string_parsers_reject_unknown_values(void) {
    TkmTokenizerKind tokenizer;
    TkmEmbedderKind embedder;
    TkmSequenceLayoutKind sequence_layout;
    TkmModelKind model;
    TkmHeadKind head;
    TkmDecoderKind decoder;
    TkmConfiguredLossKind loss;
    TkmIni ini;
    TkmLayerConfig config;

    CHECK(tkm_tokenizer_kind_from_string("int_bins", &tokenizer) == TKM_OK);
    CHECK(tokenizer == TKM_TOKENIZER_INT_BINS);
    CHECK(tkm_tokenizer_kind_from_string("normalized_continuous", &tokenizer) == TKM_OK);
    CHECK(tokenizer == TKM_TOKENIZER_NORMALIZED_CONTINUOUS);
    CHECK(tkm_embedder_kind_from_string("lookup", &embedder) == TKM_OK);
    CHECK(embedder == TKM_EMBEDDER_LOOKUP);
    CHECK(tkm_sequence_layout_kind_from_string("obs_action_interleaved", &sequence_layout) == TKM_OK);
    CHECK(sequence_layout == TKM_SEQUENCE_LAYOUT_OBS_ACTION_INTERLEAVED);
    CHECK(tkm_model_kind_from_string("lookup_policy", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_LOOKUP_POLICY);
    CHECK(tkm_model_kind_from_string("ngram", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_NGRAM);
    CHECK(tkm_model_kind_from_string("sparse_lookup", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_SPARSE_LOOKUP);
    CHECK(tkm_model_kind_from_string("nearest_policy", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_NEAREST_POLICY);
    CHECK(tkm_model_kind_from_string("linear_policy", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_LINEAR_POLICY);
    CHECK(tkm_model_kind_from_string("action_scorer", &model) == TKM_OK);
    CHECK(model == TKM_MODEL_ACTION_SCORER);
    CHECK(tkm_head_kind_from_string("categorical", &head) == TKM_OK);
    CHECK(head == TKM_HEAD_CATEGORICAL);
    CHECK(tkm_decoder_kind_from_string("argmax", &decoder) == TKM_OK);
    CHECK(decoder == TKM_DECODER_ARGMAX);
    CHECK(tkm_loss_kind_from_string("weighted_sum", &loss) == TKM_OK);
    CHECK(loss == TKM_LOSS_WEIGHTED_SUM);
    CHECK(tkm_loss_kind_from_string("action_smoothness", &loss) == TKM_OK);
    CHECK(loss == TKM_LOSS_ACTION_SMOOTHNESS);

    CHECK(tkm_tokenizer_kind_from_string("unknown", &tokenizer) == TKM_ERR);
    CHECK(tkm_embedder_kind_from_string("unknown", &embedder) == TKM_ERR);
    CHECK(tkm_sequence_layout_kind_from_string("unknown", &sequence_layout) == TKM_ERR);
    CHECK(tkm_model_kind_from_string("unknown", &model) == TKM_ERR);
    CHECK(tkm_head_kind_from_string("unknown", &head) == TKM_ERR);
    CHECK(tkm_decoder_kind_from_string("unknown", &decoder) == TKM_ERR);
    CHECK(tkm_loss_kind_from_string("unknown", &loss) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[loss]\nkind = unknown\n") == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\nhidden_dim = 0\n") == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\nhidden_dim = many\n") == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\naction_features = unknown\n") == TKM_OK);
    CHECK(tkm_layer_config_from_ini(&ini, &config) == TKM_ERR);
}

int main(void) {
    test_layer_config_reads_known_kinds_from_ini();
    test_layer_config_uses_phase0_defaults();
    test_layer_kind_string_parsers_reject_unknown_values();
    puts("layer config tests passed");
    return 0;
}
