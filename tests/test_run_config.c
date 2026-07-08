#include <stdio.h>
#include <stdlib.h>

#include "core/config/run_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_run_config_reads_project_model_and_train_sections(void) {
    const char* text =
        "[project]\n"
        "name = snake\n"
        "[model]\n"
        "kind = ngram\n"
        "hidden_dim = 64\n"
        "[train]\n"
        "loss = cross_entropy\n"
        "rollout_steps = 128\n"
        "dagger_rounds = 6\n"
        "learning_rate = 0.02\n"
        "dagger_learning_rate = 0.006\n"
        "[mask]\n"
        "strategy = none\n"
        "[decoder]\n"
        "fallback = none\n"
        "rollout_horizon = 24\n";
    TkmIni ini;
    TkmRunConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_run_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.project.project == TKM_PROJECT_SNAKE);
    CHECK(config.layers.model == TKM_MODEL_NGRAM);
    CHECK(config.layers.model_hidden_dim == 64);
    CHECK(config.train.loss == TKM_TRAIN_LOSS_CROSS_ENTROPY);
    CHECK(config.train.rollout_steps == 128);
    CHECK(config.train.dagger_rounds == 6);
    CHECK(config.train.learning_rate > 0.019f && config.train.learning_rate < 0.021f);
    CHECK(config.train.dagger_learning_rate > 0.005f && config.train.dagger_learning_rate < 0.007f);
    CHECK(config.mask.strategy == TKM_MASK_STRATEGY_NONE);
    CHECK(config.decoder.fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(config.decoder.rollout_horizon == 24);
}

static void test_run_config_reads_project_local_ini_file(void) {
    TkmRunConfig config;

    CHECK(tkm_run_config_from_file("projects/snake/token_policy.ini", &config) == TKM_OK);
    CHECK(config.project.project == TKM_PROJECT_SNAKE);
    CHECK(config.layers.model == TKM_MODEL_NGRAM);
    CHECK(config.layers.model_hidden_dim == 32);
    CHECK(config.train.loss == TKM_TRAIN_LOSS_CROSS_ENTROPY);
    CHECK(config.train.rollout_steps == 96);
    CHECK(config.train.dagger_rounds == 4);
    CHECK(config.train.learning_rate > 0.009f && config.train.learning_rate < 0.011f);
    CHECK(config.train.dagger_learning_rate > 0.004f && config.train.dagger_learning_rate < 0.006f);
    CHECK(config.mask.strategy == TKM_MASK_STRATEGY_NONE);
    CHECK(config.decoder.fallback == TKM_DECODER_FALLBACK_NONE);
    CHECK(config.decoder.rollout_horizon == 32);
}

static void test_run_config_rejects_bad_inputs(void) {
    TkmIni ini;
    TkmRunConfig config;

    CHECK(tkm_run_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[model]\nkind = unknown\n") == TKM_OK);
    CHECK(tkm_run_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_run_config_from_ini(&ini, 0) == TKM_ERR);
    CHECK(tkm_run_config_from_file("projects/snake/missing.ini", &config) == TKM_ERR);
    CHECK(tkm_run_config_from_file(0, &config) == TKM_ERR);
    CHECK(tkm_run_config_from_file("projects/snake/token_policy.ini", 0) == TKM_ERR);
}

int main(void) {
    test_run_config_reads_project_model_and_train_sections();
    test_run_config_reads_project_local_ini_file();
    test_run_config_rejects_bad_inputs();
    puts("run config tests passed");
    return 0;
}
