#include "core/config/project_config.h"

#include <string.h>

static int tkm_project_config_match(const char* text, const char* expected) {
    return text && strcmp(text, expected) == 0;
}

int tkm_project_kind_from_string(const char* text, TkmProjectKind* out) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (tkm_project_config_match(text, "centerline")) {
        *out = TKM_PROJECT_CENTERLINE;
        return TKM_OK;
    }
    if (tkm_project_config_match(text, "breakout")) {
        *out = TKM_PROJECT_BREAKOUT;
        return TKM_OK;
    }
    if (tkm_project_config_match(text, "snake")) {
        *out = TKM_PROJECT_SNAKE;
        return TKM_OK;
    }
    return TKM_ERR;
}

int tkm_project_config_from_ini(const TkmIni* ini, TkmProjectConfig* out) {
    const char* project;

    if (!ini || !out) {
        return TKM_ERR;
    }

    project = tkm_ini_get(ini, "project", "name", "");
    return tkm_project_kind_from_string(project, &out->project);
}
