#include "core/params/params.h"

#include <stdlib.h>
#include <string.h>

static int tkm_params_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static char* tkm_params_trim(char* text) {
    char* end;

    while (tkm_params_is_space(*text)) {
        text++;
    }

    end = text + strlen(text);
    while (end > text && tkm_params_is_space(*(end - 1))) {
        end--;
    }
    *end = '\0';

    return text;
}

static void tkm_params_strip_comment(char* line) {
    for (uint32_t i = 0; line[i] != '\0'; i++) {
        if (line[i] == '#' || line[i] == ';') {
            line[i] = '\0';
            return;
        }
    }
}

static int tkm_params_copy_name(char* dst, const char* src) {
    uint32_t i = 0;

    if (!dst || !src || src[0] == '\0') {
        return TKM_ERR;
    }

    while (src[i] != '\0') {
        if (i + 1 >= TKM_PARAMS_MAX_NAME) {
            return TKM_ERR;
        }
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
    return TKM_OK;
}

void tkm_params_init(TkmParams* params) {
    if (!params) {
        return;
    }

    params->array_count = 0;
    params->value_count = 0;
}

int tkm_params_get_array(const TkmParams* params, const char* name, const float** out_values, uint32_t* out_count) {
    if (!params || !name || !out_values || !out_count) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < params->array_count; i++) {
        const TkmParamArray* array = &params->arrays[i];
        if (strcmp(array->name, name) == 0) {
            *out_values = &params->values[array->offset];
            *out_count = array->count;
            return TKM_OK;
        }
    }

    return TKM_ERR;
}

int tkm_params_add_array(TkmParams* params, const char* name, const float* values, uint32_t count) {
    TkmParamArray* array;
    const float* existing_values = 0;
    uint32_t existing_count = 0;

    if (!params || !name || !values || count == 0) {
        return TKM_ERR;
    }
    if (params->array_count >= TKM_PARAMS_MAX_ARRAYS || params->value_count + count > TKM_PARAMS_MAX_VALUES) {
        return TKM_ERR;
    }
    if (tkm_params_get_array(params, name, &existing_values, &existing_count) == TKM_OK) {
        return TKM_ERR;
    }

    array = &params->arrays[params->array_count++];
    if (tkm_params_copy_name(array->name, name) != TKM_OK) {
        params->array_count--;
        return TKM_ERR;
    }
    array->offset = params->value_count;
    array->count = count;

    for (uint32_t i = 0; i < count; i++) {
        params->values[params->value_count++] = values[i];
    }

    return TKM_OK;
}

static int tkm_params_parse_values(char* value_text, float* out_values, uint32_t* out_count) {
    char* cursor = value_text;
    uint32_t count = 0;

    if (!value_text || !out_values || !out_count) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; value_text[i] != '\0'; i++) {
        if (value_text[i] == ',') {
            value_text[i] = ' ';
        }
    }

    while (*cursor != '\0') {
        char* end = 0;
        float value;

        while (tkm_params_is_space(*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }

        value = strtof(cursor, &end);
        if (end == cursor) {
            return TKM_ERR;
        }
        if (count >= TKM_PARAMS_MAX_VALUES) {
            return TKM_ERR;
        }
        out_values[count++] = value;
        cursor = end;
    }

    if (count == 0) {
        return TKM_ERR;
    }

    *out_count = count;
    return TKM_OK;
}

static int tkm_params_parse_line(TkmParams* params, char* line) {
    char* trimmed = tkm_params_trim(line);
    char* equals;
    char* name;
    char* value_text;
    float values[TKM_PARAMS_MAX_VALUES];
    uint32_t count = 0;

    if (*trimmed == '\0') {
        return TKM_OK;
    }

    equals = strchr(trimmed, '=');
    if (!equals) {
        return TKM_ERR;
    }
    *equals = '\0';

    name = tkm_params_trim(trimmed);
    value_text = tkm_params_trim(equals + 1);

    if (tkm_params_parse_values(value_text, values, &count) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_params_add_array(params, name, values, count);
}

int tkm_params_parse_text(TkmParams* params, const char* text) {
    char line[TKM_PARAMS_MAX_LINE];
    uint32_t line_len = 0;

    if (!params || !text) {
        return TKM_ERR;
    }

    tkm_params_init(params);

    for (uint32_t i = 0;; i++) {
        char c = text[i];
        if (c == '\n' || c == '\0') {
            line[line_len] = '\0';
            tkm_params_strip_comment(line);
            if (tkm_params_parse_line(params, line) != TKM_OK) {
                return TKM_ERR;
            }
            line_len = 0;
            if (c == '\0') {
                break;
            }
        } else {
            if (line_len + 1 >= TKM_PARAMS_MAX_LINE) {
                return TKM_ERR;
            }
            line[line_len++] = c;
        }
    }

    return TKM_OK;
}
