#ifndef TKM_LAYER_FACTORY_H
#define TKM_LAYER_FACTORY_H

#include "core/config/ini.h"
#include "core/embedder/embedder.h"
#include "core/model/action_scorer.h"
#include "core/model/mlp_window.h"
#include "core/model/ngram.h"
#include "core/params/params.h"
#include "core/tokenizer/int_bins.h"
#include "core/tokenizer/vq_code.h"
#include "core/vectorizer/continuous.h"

int tkm_layer_factory_init_int_bins(const TkmIni* ini, TkmIntBins* out);
int tkm_layer_factory_init_raw_continuous(const TkmIni* ini, TkmContinuousVectorizer* out);
int tkm_layer_factory_init_normalized_continuous(
    const TkmIni* ini,
    const TkmParams* params,
    TkmContinuousVectorizer* out
);
int tkm_layer_factory_init_ngram(const TkmIni* ini, TkmNgramModel* out);
int tkm_layer_factory_init_vq_code(const TkmIni* ini, const TkmParams* params, TkmVqCode* out);
int tkm_layer_factory_init_lookup_embedder(const TkmIni* ini, const TkmParams* params, TkmLookupEmbedder* out);
int tkm_layer_factory_init_linear_embedder(const TkmIni* ini, const TkmParams* params, TkmLinearEmbedder* out);
int tkm_layer_factory_init_mlp_window(const TkmIni* ini, const TkmParams* params, TkmMlpWindow* out);
int tkm_layer_factory_init_action_scorer(const TkmIni* ini, const TkmParams* params, TkmActionScorer* out);

#endif
