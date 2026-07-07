#include <stdio.h>
#include <stdlib.h>

#include "core/config/project_config.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void test_project_config_reads_project_name_from_ini(void) {
    TkmIni ini;
    TkmProjectConfig config;

    CHECK(tkm_ini_parse(&ini, "[project]\nname = snake\n") == TKM_OK);
    CHECK(tkm_project_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.project == TKM_PROJECT_SNAKE);
}

static void test_project_kind_parser_accepts_current_projects(void) {
    TkmProjectKind project;

    CHECK(tkm_project_kind_from_string("centerline", &project) == TKM_OK);
    CHECK(project == TKM_PROJECT_CENTERLINE);
    CHECK(tkm_project_kind_from_string("breakout", &project) == TKM_OK);
    CHECK(project == TKM_PROJECT_BREAKOUT);
    CHECK(tkm_project_kind_from_string("snake", &project) == TKM_OK);
    CHECK(project == TKM_PROJECT_SNAKE);
}

static void test_project_config_rejects_missing_or_unknown_project(void) {
    TkmIni ini;
    TkmProjectConfig config;
    TkmProjectKind project;

    CHECK(tkm_project_kind_from_string("unknown", &project) == TKM_ERR);
    CHECK(tkm_project_kind_from_string(0, &project) == TKM_ERR);
    CHECK(tkm_project_kind_from_string("snake", 0) == TKM_ERR);

    CHECK(tkm_ini_parse(&ini, "") == TKM_OK);
    CHECK(tkm_project_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_ini_parse(&ini, "[project]\nname = unknown\n") == TKM_OK);
    CHECK(tkm_project_config_from_ini(&ini, &config) == TKM_ERR);
    CHECK(tkm_project_config_from_ini(0, &config) == TKM_ERR);
    CHECK(tkm_project_config_from_ini(&ini, 0) == TKM_ERR);
}

int main(void) {
    test_project_config_reads_project_name_from_ini();
    test_project_kind_parser_accepts_current_projects();
    test_project_config_rejects_missing_or_unknown_project();
    puts("project config tests passed");
    return 0;
}
