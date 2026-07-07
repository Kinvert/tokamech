#include <stdio.h>
#include <stdlib.h>

#include "core/config/train_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_train_config_reads_loss_from_ini(void) {
    const char* text =
        "[train]\n"
        "loss = masked_cross_entropy\n"
        "rollout_steps = 240\n"
        "dagger_rounds = 6\n"
        "learning_rate = 0.02\n"
        "dagger_learning_rate = 0.006\n";
    TkmIni ini;
    TkmTrainConfig config;

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY);
    CHECK(config.rollout_steps == 240);
    CHECK(config.dagger_rounds == 6);
    CHECK(config.learning_rate > 0.019f && config.learning_rate < 0.021f);
    CHECK(config.dagger_learning_rate > 0.005f && config.dagger_learning_rate < 0.007f);
}

static void test_train_config_uses_simple_defaults(void) {
    TkmIni ini;
    TkmTrainConfig config;

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.loss == TKM_TRAIN_LOSS_CROSS_ENTROPY);
    CHECK(config.rollout_steps == 96);
    CHECK(config.dagger_rounds == 4);
    CHECK(config.learning_rate == 0.0f);
    CHECK(config.dagger_learning_rate > 0.004f && config.dagger_learning_rate < 0.006f);
}

static void test_train_loss_parser_accepts_interchangeable_options(void) {
    TkmTrainLossKind loss;
    TkmIni ini;
    TkmTrainConfig config;

    CHECK(tkm_train_loss_kind_from_string("cross_entropy", &loss) == TKM_OK);
    CHECK(loss == TKM_TRAIN_LOSS_CROSS_ENTROPY);
    CHECK(tkm_train_loss_kind_from_string("masked_cross_entropy", &loss) == TKM_OK);
    CHECK(loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY);
    CHECK(tkm_train_loss_kind_from_string("mse", &loss) == TKM_OK);
    CHECK(loss == TKM_TRAIN_LOSS_MSE);
    CHECK(tkm_train_loss_kind_from_string("margin", &loss) == TKM_OK);
    CHECK(loss == TKM_TRAIN_LOSS_MARGIN);

    CHECK(tkm_train_loss_kind_from_string(0, &loss) == TKM_ERR);
    CHECK(tkm_train_loss_kind_from_string("unknown", &loss) == TKM_ERR);
    CHECK(tkm_train_loss_kind_from_string("cross_entropy", 0) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "[train]\nloss = unknown\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\nrollout_steps = 0\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\nrollout_steps = many\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\ndagger_rounds = 0\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\ndagger_rounds = many\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\nlearning_rate = 0\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\nlearning_rate = many\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\ndagger_learning_rate = 0\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[train]\ndagger_learning_rate = many\n") == TKM_OK);
    CHECK(tkm_train_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_train_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_train_config_from_ini(&ini, 0) == TKM_ERR);
}

int main(void) {
    test_train_config_reads_loss_from_ini();
    test_train_config_uses_simple_defaults();
    test_train_loss_parser_accepts_interchangeable_options();
    puts("train config tests passed");
    return 0;
}
