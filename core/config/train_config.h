#ifndef TKM_TRAIN_CONFIG_H
#define TKM_TRAIN_CONFIG_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/config/ini.h"

#define TKM_TRAIN_DEFAULT_ROLLOUT_STEPS 96u
#define TKM_TRAIN_MAX_ROLLOUT_STEPS 4096u
#define TKM_TRAIN_DEFAULT_DAGGER_ROUNDS 4u
#define TKM_TRAIN_MAX_DAGGER_ROUNDS 64u
#define TKM_TRAIN_AUTO_LEARNING_RATE 0.0f
#define TKM_TRAIN_DEFAULT_DAGGER_LEARNING_RATE 0.005f
#define TKM_TRAIN_MAX_LEARNING_RATE 1.0f

typedef enum {
    TKM_TRAIN_LOSS_CROSS_ENTROPY = 0,
    TKM_TRAIN_LOSS_MSE = 1,
    TKM_TRAIN_LOSS_MARGIN = 2,
} TkmTrainLossKind;

typedef struct {
    TkmTrainLossKind loss;
    uint32_t rollout_steps;
    uint32_t dagger_rounds;
    float learning_rate;
    float dagger_learning_rate;
} TkmTrainConfig;

int tkm_train_loss_kind_from_string(const char* text, TkmTrainLossKind* out);
int tkm_train_config_from_ini(const TkmIni* ini, TkmTrainConfig* out);

#endif
