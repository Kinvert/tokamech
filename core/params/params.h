#ifndef TKM_PARAMS_H
#define TKM_PARAMS_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_PARAMS_MAX_ARRAYS 64
#define TKM_PARAMS_MAX_VALUES 4096
#define TKM_PARAMS_MAX_NAME 32
#define TKM_PARAMS_MAX_LINE 512

typedef struct {
    char name[TKM_PARAMS_MAX_NAME];
    uint32_t offset;
    uint32_t count;
} TkmParamArray;

typedef struct {
    uint32_t array_count;
    uint32_t value_count;
    TkmParamArray arrays[TKM_PARAMS_MAX_ARRAYS];
    float values[TKM_PARAMS_MAX_VALUES];
} TkmParams;

void tkm_params_init(TkmParams* params);
int tkm_params_add_array(TkmParams* params, const char* name, const float* values, uint32_t count);
int tkm_params_get_array(const TkmParams* params, const char* name, const float** out_values, uint32_t* out_count);
int tkm_params_parse_text(TkmParams* params, const char* text);

#endif
