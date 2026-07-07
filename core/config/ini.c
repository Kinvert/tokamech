#include "core/config/ini.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tkm_ini_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static char* tkm_ini_trim(char* text) {
    char* end;

    while (tkm_ini_is_space(*text)) {
        text++;
    }

    end = text + strlen(text);
    while (end > text && tkm_ini_is_space(*(end - 1))) {
        end--;
    }
    *end = '\0';

    return text;
}

static int tkm_ini_copy(char* dst, uint32_t dst_count, const char* src) {
    uint32_t i = 0;

    if (!dst || !src || dst_count == 0) {
        return TKM_ERR;
    }

    while (src[i] != '\0') {
        if (i + 1 >= dst_count) {
            return TKM_ERR;
        }
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
    return TKM_OK;
}

static TkmIniEntry* tkm_ini_add_entry(TkmIni* ini) {
    if (ini->count >= TKM_INI_MAX_ENTRIES) {
        return 0;
    }
    return &ini->entries[ini->count++];
}

static const TkmIniEntry* tkm_ini_find(const TkmIni* ini, const char* section, const char* key) {
    if (!ini || !section || !key) {
        return 0;
    }

    for (uint32_t i = ini->count; i > 0; i--) {
        const TkmIniEntry* entry = &ini->entries[i - 1];
        if (strcmp(entry->section, section) == 0 && strcmp(entry->key, key) == 0) {
            return entry;
        }
    }

    return 0;
}

static void tkm_ini_strip_comment(char* line) {
    for (uint32_t i = 0; line[i] != '\0'; i++) {
        if (line[i] == '#' || line[i] == ';') {
            line[i] = '\0';
            return;
        }
    }
}

static int tkm_ini_parse_line(TkmIni* ini, char* current_section, char* line) {
    char* trimmed = tkm_ini_trim(line);
    char* equals;
    TkmIniEntry* entry;

    if (*trimmed == '\0') {
        return TKM_OK;
    }

    if (*trimmed == '[') {
        char* close = strchr(trimmed, ']');
        char* after_close;
        if (!close) {
            return TKM_ERR;
        }
        *close = '\0';
        after_close = tkm_ini_trim(close + 1);
        if (*after_close != '\0') {
            return TKM_ERR;
        }
        return tkm_ini_copy(current_section, TKM_INI_MAX_SECTION, tkm_ini_trim(trimmed + 1));
    }

    equals = strchr(trimmed, '=');
    if (!equals) {
        return TKM_ERR;
    }
    *equals = '\0';

    entry = tkm_ini_add_entry(ini);
    if (!entry) {
        return TKM_ERR;
    }
    if (tkm_ini_copy(entry->section, TKM_INI_MAX_SECTION, current_section) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_ini_copy(entry->key, TKM_INI_MAX_KEY, tkm_ini_trim(trimmed)) != TKM_OK) {
        return TKM_ERR;
    }
    if (tkm_ini_copy(entry->value, TKM_INI_MAX_VALUE, tkm_ini_trim(equals + 1)) != TKM_OK) {
        return TKM_ERR;
    }
    if (entry->key[0] == '\0') {
        return TKM_ERR;
    }

    return TKM_OK;
}

int tkm_ini_parse(TkmIni* ini, const char* text) {
    char current_section[TKM_INI_MAX_SECTION] = "";
    char line[TKM_INI_MAX_LINE];
    uint32_t line_len = 0;

    if (!ini || !text) {
        return TKM_ERR;
    }

    ini->count = 0;

    for (uint32_t i = 0;; i++) {
        char c = text[i];
        if (c == '\n' || c == '\0') {
            line[line_len] = '\0';
            tkm_ini_strip_comment(line);
            if (tkm_ini_parse_line(ini, current_section, line) != TKM_OK) {
                return TKM_ERR;
            }
            line_len = 0;
            if (c == '\0') {
                break;
            }
        } else {
            if (line_len + 1 >= TKM_INI_MAX_LINE) {
                return TKM_ERR;
            }
            line[line_len++] = c;
        }
    }

    return TKM_OK;
}

int tkm_ini_parse_file(TkmIni* ini, const char* path) {
    FILE* file;
    char text[TKM_INI_MAX_FILE_BYTES + 1];
    size_t bytes_read;

    if (!ini || !path) {
        return TKM_ERR;
    }

    file = fopen(path, "rb");
    if (!file) {
        return TKM_ERR;
    }

    bytes_read = fread(text, 1u, TKM_INI_MAX_FILE_BYTES, file);
    if (ferror(file) || (!feof(file) && bytes_read == TKM_INI_MAX_FILE_BYTES)) {
        fclose(file);
        return TKM_ERR;
    }
    if (fclose(file) != 0) {
        return TKM_ERR;
    }

    text[bytes_read] = '\0';
    return tkm_ini_parse(ini, text);
}

const char* tkm_ini_get(const TkmIni* ini, const char* section, const char* key, const char* default_value) {
    const TkmIniEntry* entry = tkm_ini_find(ini, section, key);
    if (!entry) {
        return default_value;
    }
    return entry->value;
}

int tkm_ini_get_i32(const TkmIni* ini, const char* section, const char* key, int32_t default_value, int32_t* out) {
    const TkmIniEntry* entry;
    const char* text;
    char* end = 0;
    long value;

    if (!out) {
        return TKM_ERR;
    }

    entry = tkm_ini_find(ini, section, key);
    if (!entry) {
        *out = default_value;
        return TKM_OK;
    }

    text = entry->value;
    value = strtol(text, &end, 10);
    if (end == text || *tkm_ini_trim(end) != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return TKM_ERR;
    }

    *out = (int32_t)value;
    return TKM_OK;
}

int tkm_ini_get_f32(const TkmIni* ini, const char* section, const char* key, float default_value, float* out) {
    const TkmIniEntry* entry;
    const char* text;
    char* end = 0;
    float value;

    if (!out) {
        return TKM_ERR;
    }

    entry = tkm_ini_find(ini, section, key);
    if (!entry) {
        *out = default_value;
        return TKM_OK;
    }

    text = entry->value;
    value = strtof(text, &end);
    if (end == text || *tkm_ini_trim(end) != '\0') {
        return TKM_ERR;
    }

    *out = value;
    return TKM_OK;
}
