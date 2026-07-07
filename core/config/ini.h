#ifndef TKM_INI_H
#define TKM_INI_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_INI_MAX_ENTRIES 128
#define TKM_INI_MAX_SECTION 32
#define TKM_INI_MAX_KEY 32
#define TKM_INI_MAX_VALUE 96
#define TKM_INI_MAX_LINE 192
#define TKM_INI_MAX_FILE_BYTES 4096

typedef struct {
    char section[TKM_INI_MAX_SECTION];
    char key[TKM_INI_MAX_KEY];
    char value[TKM_INI_MAX_VALUE];
} TkmIniEntry;

typedef struct {
    uint32_t count;
    TkmIniEntry entries[TKM_INI_MAX_ENTRIES];
} TkmIni;

int tkm_ini_parse(TkmIni* ini, const char* text);
int tkm_ini_parse_file(TkmIni* ini, const char* path);
const char* tkm_ini_get(const TkmIni* ini, const char* section, const char* key, const char* default_value);
int tkm_ini_get_i32(const TkmIni* ini, const char* section, const char* key, int32_t default_value, int32_t* out);
int tkm_ini_get_f32(const TkmIni* ini, const char* section, const char* key, float default_value, float* out);

#endif
