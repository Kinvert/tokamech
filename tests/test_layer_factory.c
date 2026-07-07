#include <stdio.h>
#include <stdlib.h>

#include "core/config/layer_factory.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void check_close(float actual, float expected) {
    float diff = actual - expected;
    if (diff < 0.0f) {
        diff = -diff;
    }
    CHECK(diff < 0.0001f);
}

static void test_factory_constructs_int_bins_from_ini(void) {
    const char* text =
        "[tokenizer]\n"
        "kind = int_bins\n"
        "min = -4\n"
        "max = 4\n";
    TkmIni ini;
    TkmIntBins bins;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_layer_factory_init_int_bins(&ini, &bins) == TKM_OK);
    CHECK(bins.min_value == -4);
    CHECK(bins.max_value == 4);
    CHECK(bins.vocab_size == 9);
    CHECK(tkm_int_bins_encode(&bins, -4) == 0);
    CHECK(tkm_int_bins_encode(&bins, 4) == 8);
}

static void test_factory_constructs_raw_continuous_vectorizer_from_ini(void) {
    const char* text =
        "[tokenizer]\n"
        "kind = raw_continuous\n"
        "dim = 3\n";
    const float input[3] = {1.0f, -2.0f, 3.0f};
    TkmIni ini;
    TkmContinuousVectorizer vectorizer;
    float output[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_layer_factory_init_raw_continuous(&ini, &vectorizer) == TKM_OK);
    CHECK(vectorizer.kind == TKM_CONTINUOUS_RAW);
    CHECK(vectorizer.dim == 3);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, output, 3) == TKM_OK);
    check_close(output[0], 1.0f);
    check_close(output[1], -2.0f);
    check_close(output[2], 3.0f);
}

static void test_factory_constructs_ngram_from_ini(void) {
    const char* text =
        "[model]\n"
        "kind = ngram\n"
        "vocab_size = 200\n";
    TkmIni ini;
    TkmNgramModel model;
    uint16_t next = 99;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_layer_factory_init_ngram(&ini, &model) == TKM_OK);
    CHECK(model.vocab_size == 200);
    CHECK(tkm_ngram_predict_next(&model, 10, &next) == TKM_OK);
    CHECK(next == 0);
}

static void test_factory_constructs_vq_code_from_ini_and_params(void) {
    const char* ini_text =
        "[tokenizer]\n"
        "kind = vq_code\n"
        "dim = 2\n"
        "codes = 3\n"
        "codebook = toy_codebook\n";
    const char* param_text = "toy_codebook = 0.0 0.0 1.0 0.0 0.0 1.0\n";
    const float vector[2] = {0.9f, 0.1f};
    TkmIni ini;
    TkmParams params;
    TkmVqCode vq;
    uint16_t code = 99;
    float distance = 0.0f;

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_vq_code(&ini, &params, &vq) == TKM_OK);
    CHECK(vq.code_count == 3);
    CHECK(vq.dim == 2);
    CHECK(tkm_vq_code_encode(&vq, vector, &code, &distance) == TKM_OK);
    CHECK(code == 1);
    check_close(distance, 0.02f);
}

static void test_factory_constructs_normalized_continuous_from_ini_and_params(void) {
    const char* ini_text =
        "[tokenizer]\n"
        "kind = normalized_continuous\n"
        "dim = 2\n"
        "mean = qti_mean\n"
        "std = qti_std\n";
    const char* param_text =
        "qti_mean = 10.0 0.0\n"
        "qti_std = 2.0 4.0\n";
    const float input[2] = {14.0f, -4.0f};
    TkmIni ini;
    TkmParams params;
    TkmContinuousVectorizer vectorizer;
    float output[2] = {0.0f, 0.0f};

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_normalized_continuous(&ini, &params, &vectorizer) == TKM_OK);
    CHECK(vectorizer.kind == TKM_CONTINUOUS_STANDARDIZE);
    CHECK(vectorizer.dim == 2);
    CHECK(tkm_continuous_vectorizer_apply(&vectorizer, input, output, 2) == TKM_OK);
    check_close(output[0], 2.0f);
    check_close(output[1], -1.0f);
}

static void test_factory_constructs_lookup_embedder_from_ini_and_params(void) {
    const char* ini_text =
        "[embedder]\n"
        "kind = lookup\n"
        "tokens = 3\n"
        "dim = 2\n"
        "table = token_table\n";
    const char* param_text = "token_table = 1.0 2.0 3.0 4.0 5.0 6.0\n";
    TkmIni ini;
    TkmParams params;
    TkmLookupEmbedder embedder;
    float output[2] = {0.0f, 0.0f};

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_lookup_embedder(&ini, &params, &embedder) == TKM_OK);
    CHECK(embedder.token_count == 3);
    CHECK(embedder.dim == 2);
    CHECK(tkm_lookup_embedder_encode(&embedder, 2, output, 2) == TKM_OK);
    check_close(output[0], 5.0f);
    check_close(output[1], 6.0f);
}

static void test_factory_constructs_linear_embedder_from_ini_and_params(void) {
    const char* ini_text =
        "[embedder]\n"
        "kind = linear\n"
        "input_dim = 2\n"
        "output_dim = 3\n"
        "weights = linear_w\n"
        "bias = linear_b\n";
    const char* param_text =
        "linear_w = 1.0 0.0 2.0 -1.0 1.0 0.5\n"
        "linear_b = 0.5 -0.5 1.0\n";
    const float input[2] = {2.0f, 3.0f};
    TkmIni ini;
    TkmParams params;
    TkmLinearEmbedder embedder;
    float output[3] = {0.0f, 0.0f, 0.0f};

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_linear_embedder(&ini, &params, &embedder) == TKM_OK);
    CHECK(embedder.input_dim == 2);
    CHECK(embedder.output_dim == 3);
    CHECK(tkm_linear_embedder_encode(&embedder, input, output, 3) == TKM_OK);
    check_close(output[0], -0.5f);
    check_close(output[1], 2.5f);
    check_close(output[2], 6.5f);
}

static void test_factory_constructs_mlp_window_from_ini_and_params(void) {
    const char* ini_text =
        "[model]\n"
        "kind = mlp_window\n"
        "input_dim = 2\n"
        "hidden_dim = 3\n"
        "output_dim = 2\n"
        "w0 = mlp_w0\n"
        "b0 = mlp_b0\n"
        "w1 = mlp_w1\n"
        "b1 = mlp_b1\n";
    const char* param_text =
        "mlp_w0 = 1.0 -1.0 0.5 0.0 2.0 -1.0\n"
        "mlp_b0 = 0.0 -1.0 0.5\n"
        "mlp_w1 = 1.0 0.0 0.5 1.0 -1.0 2.0\n"
        "mlp_b1 = 0.25 -0.25\n";
    const float input[2] = {2.0f, 3.0f};
    TkmIni ini;
    TkmParams params;
    TkmMlpWindow mlp;
    float output[2] = {0.0f, 0.0f};

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_mlp_window(&ini, &params, &mlp) == TKM_OK);
    CHECK(mlp.input_dim == 2);
    CHECK(mlp.hidden_dim == 3);
    CHECK(mlp.output_dim == 2);
    CHECK(tkm_mlp_window_forward(&mlp, input, output, 2) == TKM_OK);
    check_close(output[0], 3.75f);
    check_close(output[1], 2.75f);
}

static void test_factory_constructs_action_scorer_from_ini_and_params(void) {
    const char* ini_text =
        "[model]\n"
        "kind = action_scorer\n"
        "state_dim = 1\n"
        "action_dim = 2\n"
        "hidden_dim = 1\n"
        "w0 = scorer_w0\n"
        "b0 = scorer_b0\n"
        "w1 = scorer_w1\n"
        "b1 = scorer_b1\n";
    const char* param_text =
        "scorer_w0 = 0.0 1.0 1.0\n"
        "scorer_b0 = 0.0\n"
        "scorer_w1 = 1.0\n"
        "scorer_b1 = 0.0\n";
    const float state[1] = {0.0f};
    const float action[2] = {0.0f, 3.0f};
    TkmIni ini;
    TkmParams params;
    TkmActionScorer scorer;
    float score = 0.0f;

    CHECK(tkm_ini_parse(&ini, ini_text) == TKM_OK);
    CHECK(tkm_params_parse_text(&params, param_text) == TKM_OK);
    CHECK(tkm_layer_factory_init_action_scorer(&ini, &params, &scorer) == TKM_OK);
    CHECK(scorer.state_dim == 1);
    CHECK(scorer.action_dim == 2);
    CHECK(scorer.mlp.hidden_dim == 1);
    CHECK(tkm_action_scorer_score(&scorer, state, action, &score) == TKM_OK);
    check_close(score, 3.0f);
}

static void test_factory_rejects_missing_required_fields_and_wrong_kinds(void) {
    TkmIni ini;
    TkmParams params;
    TkmIntBins bins;
    TkmContinuousVectorizer vectorizer;
    TkmNgramModel model;
    TkmVqCode vq;
    TkmLookupEmbedder lookup_embedder;
    TkmLinearEmbedder linear_embedder;
    TkmMlpWindow mlp;
    TkmActionScorer scorer;

    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = int_bins\nmin = -4\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_int_bins(&ini, &bins) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = vq_code\ncodes = 4\ndim = 2\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_int_bins(&ini, &bins) == TKM_ERR);
    CHECK(tkm_layer_factory_init_raw_continuous(&ini, &vectorizer) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = raw_continuous\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_raw_continuous(&ini, &vectorizer) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[model]\nkind = mlp_window\ninput_dim = 2\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_ngram(&ini, &model) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[model]\nkind = ngram\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_ngram(&ini, &model) == TKM_ERR);

    CHECK(tkm_params_parse_text(&params, "short = 0.0 1.0\n") == TKM_OK);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = vq_code\ndim = 2\ncodes = 3\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_vq_code(&ini, &params, &vq) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = vq_code\ndim = 2\ncodes = 3\ncodebook = missing\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_vq_code(&ini, &params, &vq) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = vq_code\ndim = 2\ncodes = 3\ncodebook = short\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_vq_code(&ini, &params, &vq) == TKM_ERR);

    CHECK(tkm_params_parse_text(&params, "mean = 0.0 1.0\nshort_std = 1.0\n") == TKM_OK);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = normalized_continuous\ndim = 2\nmean = mean\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_normalized_continuous(&ini, &params, &vectorizer) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = normalized_continuous\ndim = 2\nmean = missing\nstd = short_std\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_normalized_continuous(&ini, &params, &vectorizer) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[tokenizer]\nkind = normalized_continuous\ndim = 2\nmean = mean\nstd = short_std\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_normalized_continuous(&ini, &params, &vectorizer) == TKM_ERR);

    CHECK(tkm_params_parse_text(&params, "table = 1.0 2.0\nweights = 1.0 2.0\nbias = 0.0\n") == TKM_OK);
    CHECK(tkm_ini_parse(&ini, "[embedder]\nkind = lookup\ntokens = 2\ndim = 2\ntable = table\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_lookup_embedder(&ini, &params, &lookup_embedder) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[embedder]\nkind = linear\ninput_dim = 2\noutput_dim = 2\nweights = weights\nbias = bias\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_linear_embedder(&ini, &params, &linear_embedder) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\nkind = mlp_window\ninput_dim = 2\nhidden_dim = 2\noutput_dim = 1\nw0 = weights\nb0 = bias\nw1 = weights\nb1 = bias\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_mlp_window(&ini, &params, &mlp) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\nkind = action_scorer\nstate_dim = 1\naction_dim = 2\nhidden_dim = 1\nw0 = weights\nb0 = bias\nw1 = weights\nb1 = bias\n") == TKM_OK);
    CHECK(tkm_layer_factory_init_action_scorer(&ini, &params, &scorer) == TKM_ERR);
}

int main(void) {
    test_factory_constructs_int_bins_from_ini();
    test_factory_constructs_raw_continuous_vectorizer_from_ini();
    test_factory_constructs_ngram_from_ini();
    test_factory_constructs_vq_code_from_ini_and_params();
    test_factory_constructs_normalized_continuous_from_ini_and_params();
    test_factory_constructs_lookup_embedder_from_ini_and_params();
    test_factory_constructs_linear_embedder_from_ini_and_params();
    test_factory_constructs_mlp_window_from_ini_and_params();
    test_factory_constructs_action_scorer_from_ini_and_params();
    test_factory_rejects_missing_required_fields_and_wrong_kinds();
    puts("layer factory tests passed");
    return 0;
}
