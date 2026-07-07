#ifndef TKM_PROJECT_CONFIG_H
#define TKM_PROJECT_CONFIG_H

#include "core/common/status.h"
#include "core/config/ini.h"

typedef enum {
    TKM_PROJECT_CENTERLINE = 0,
    TKM_PROJECT_BREAKOUT = 1,
    TKM_PROJECT_SNAKE = 2,
} TkmProjectKind;

typedef struct {
    TkmProjectKind project;
} TkmProjectConfig;

int tkm_project_kind_from_string(const char* text, TkmProjectKind* out);
int tkm_project_config_from_ini(const TkmIni* ini, TkmProjectConfig* out);

#endif
