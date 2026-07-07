#ifndef TKM_RUN_CONFIG_H
#define TKM_RUN_CONFIG_H

#include "core/common/status.h"
#include "core/config/decoder_config.h"
#include "core/config/ini.h"
#include "core/config/layer_config.h"
#include "core/config/mask_config.h"
#include "core/config/project_config.h"
#include "core/config/train_config.h"

typedef struct {
    TkmProjectConfig project;
    TkmLayerConfig layers;
    TkmTrainConfig train;
    TkmMaskConfig mask;
    TkmDecoderConfig decoder;
} TkmRunConfig;

int tkm_run_config_from_ini(const TkmIni* ini, TkmRunConfig* out);
int tkm_run_config_from_file(const char* path, TkmRunConfig* out);

#endif
