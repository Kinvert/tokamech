#include "core/config/layer_factory.h"

#include "core/config/layer_config.h"

static int tkm_layer_factory_required_i32(
    const TkmIni* ini,
    const char* section,
    const char* key,
    int32_t* out
) {
    if (!ini || !section || !key || !out) {
        return TKM_ERR;
    }
    if (!tkm_ini_get(ini, section, key, 0)) {
        return TKM_ERR;
    }
    return tkm_ini_get_i32(ini, section, key, 0, out);
}

static const char* tkm_layer_factory_required_string(
    const TkmIni* ini,
    const char* section,
    const char* key
) {
    const char* value;

    if (!ini || !section || !key) {
        return 0;
    }

    value = tkm_ini_get(ini, section, key, 0);
    if (!value || value[0] == '\0') {
        return 0;
    }
    return value;
}

static int tkm_layer_factory_get_param(
    const TkmParams* params,
    const char* name,
    uint32_t required_count,
    const float** out_values
) {
    uint32_t count = 0;

    if (!params || !name || !out_values || required_count == 0) {
        return TKM_ERR;
    }
    if (tkm_params_get_array(params, name, out_values, &count) != TKM_OK || count != required_count) {
        return TKM_ERR;
    }
    return TKM_OK;
}

int tkm_layer_factory_init_int_bins(const TkmIni* ini, TkmIntBins* out) {
    TkmLayerConfig config;
    int32_t min_value = 0;
    int32_t max_value = 0;

    if (!ini || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.tokenizer != TKM_TOKENIZER_INT_BINS) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "tokenizer", "min", &min_value) != TKM_OK ||
        tkm_layer_factory_required_i32(ini, "tokenizer", "max", &max_value) != TKM_OK) {
        return TKM_ERR;
    }
    return tkm_int_bins_init(out, min_value, max_value);
}

int tkm_layer_factory_init_raw_continuous(const TkmIni* ini, TkmContinuousVectorizer* out) {
    TkmLayerConfig config;
    int32_t dim = 0;

    if (!ini || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.tokenizer != TKM_TOKENIZER_RAW_CONTINUOUS) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "tokenizer", "dim", &dim) != TKM_OK || dim <= 0) {
        return TKM_ERR;
    }

    return tkm_continuous_vectorizer_init_raw(out, (uint32_t)dim);
}

int tkm_layer_factory_init_normalized_continuous(
    const TkmIni* ini,
    const TkmParams* params,
    TkmContinuousVectorizer* out
) {
    TkmLayerConfig config;
    int32_t dim = 0;
    const char* mean_name;
    const char* std_name;
    const float* mean = 0;
    const float* std = 0;
    uint32_t mean_count = 0;
    uint32_t std_count = 0;

    if (!ini || !params || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK ||
        config.tokenizer != TKM_TOKENIZER_NORMALIZED_CONTINUOUS) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "tokenizer", "dim", &dim) != TKM_OK || dim <= 0) {
        return TKM_ERR;
    }

    mean_name = tkm_layer_factory_required_string(ini, "tokenizer", "mean");
    std_name = tkm_layer_factory_required_string(ini, "tokenizer", "std");
    if (!mean_name || !std_name) {
        return TKM_ERR;
    }
    if (tkm_params_get_array(params, mean_name, &mean, &mean_count) != TKM_OK ||
        tkm_params_get_array(params, std_name, &std, &std_count) != TKM_OK) {
        return TKM_ERR;
    }
    if (mean_count != (uint32_t)dim || std_count != (uint32_t)dim) {
        return TKM_ERR;
    }

    return tkm_continuous_vectorizer_init_standardize(out, (uint32_t)dim, mean, std);
}

int tkm_layer_factory_init_ngram(const TkmIni* ini, TkmNgramModel* out) {
    TkmLayerConfig config;
    int32_t vocab_size = 0;

    if (!ini || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.model != TKM_MODEL_NGRAM) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "model", "vocab_size", &vocab_size) != TKM_OK || vocab_size <= 0) {
        return TKM_ERR;
    }

    return tkm_ngram_init(out, (uint32_t)vocab_size);
}

int tkm_layer_factory_init_vq_code(const TkmIni* ini, const TkmParams* params, TkmVqCode* out) {
    TkmLayerConfig config;
    int32_t dim = 0;
    int32_t codes = 0;
    const char* codebook_name;
    const float* codebook = 0;
    uint32_t codebook_count = 0;
    uint32_t required_count = 0;

    if (!ini || !params || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.tokenizer != TKM_TOKENIZER_VQ_CODE) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "tokenizer", "dim", &dim) != TKM_OK ||
        tkm_layer_factory_required_i32(ini, "tokenizer", "codes", &codes) != TKM_OK ||
        dim <= 0 ||
        codes <= 0) {
        return TKM_ERR;
    }

    codebook_name = tkm_layer_factory_required_string(ini, "tokenizer", "codebook");
    if (!codebook_name) {
        return TKM_ERR;
    }
    if (tkm_params_get_array(params, codebook_name, &codebook, &codebook_count) != TKM_OK) {
        return TKM_ERR;
    }

    required_count = (uint32_t)dim * (uint32_t)codes;
    if (codebook_count != required_count) {
        return TKM_ERR;
    }

    return tkm_vq_code_init(out, (uint32_t)codes, (uint32_t)dim, codebook);
}

int tkm_layer_factory_init_lookup_embedder(const TkmIni* ini, const TkmParams* params, TkmLookupEmbedder* out) {
    TkmLayerConfig config;
    int32_t tokens = 0;
    int32_t dim = 0;
    const char* table_name;
    const float* table = 0;
    uint32_t required_count = 0;

    if (!ini || !params || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.embedder != TKM_EMBEDDER_LOOKUP) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "embedder", "tokens", &tokens) != TKM_OK ||
        tkm_layer_factory_required_i32(ini, "embedder", "dim", &dim) != TKM_OK ||
        tokens <= 0 ||
        dim <= 0) {
        return TKM_ERR;
    }

    table_name = tkm_layer_factory_required_string(ini, "embedder", "table");
    required_count = (uint32_t)tokens * (uint32_t)dim;
    if (!table_name || tkm_layer_factory_get_param(params, table_name, required_count, &table) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_lookup_embedder_init(out, (uint32_t)tokens, (uint32_t)dim, table);
}

int tkm_layer_factory_init_linear_embedder(const TkmIni* ini, const TkmParams* params, TkmLinearEmbedder* out) {
    TkmLayerConfig config;
    int32_t input_dim = 0;
    int32_t output_dim = 0;
    const char* weights_name;
    const char* bias_name;
    const float* weights = 0;
    const float* bias = 0;

    if (!ini || !params || !out) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &config) != TKM_OK || config.embedder != TKM_EMBEDDER_LINEAR) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_required_i32(ini, "embedder", "input_dim", &input_dim) != TKM_OK ||
        tkm_layer_factory_required_i32(ini, "embedder", "output_dim", &output_dim) != TKM_OK ||
        input_dim <= 0 ||
        output_dim <= 0) {
        return TKM_ERR;
    }

    weights_name = tkm_layer_factory_required_string(ini, "embedder", "weights");
    bias_name = tkm_layer_factory_required_string(ini, "embedder", "bias");
    if (!weights_name || !bias_name) {
        return TKM_ERR;
    }
    if (tkm_layer_factory_get_param(params, weights_name, (uint32_t)input_dim * (uint32_t)output_dim, &weights) != TKM_OK ||
        tkm_layer_factory_get_param(params, bias_name, (uint32_t)output_dim, &bias) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_linear_embedder_init(out, (uint32_t)input_dim, (uint32_t)output_dim, weights, bias);
}
