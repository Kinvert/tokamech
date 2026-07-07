#include "core/config/run_config.h"

int tkm_run_config_from_ini(const TkmIni* ini, TkmRunConfig* out) {
    if (!ini || !out) {
        return TKM_ERR;
    }

    if (tkm_project_config_from_ini(ini, &out->project) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_layer_config_from_ini(ini, &out->layers) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_train_config_from_ini(ini, &out->train) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_mask_config_from_ini(ini, &out->mask) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_decoder_config_from_ini(ini, &out->decoder) != TKM_OK) {
        return TKM_ERR;
    }

    return TKM_OK;
}

int tkm_run_config_from_file(const char* path, TkmRunConfig* out) {
    TkmIni ini;

    if (!path || !out) {
        return TKM_ERR;
    }
    if (tkm_ini_parse_file(&ini, path) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_run_config_from_ini(&ini, out);
}
