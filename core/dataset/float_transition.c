#include "core/dataset/float_transition.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TKM_FLOAT_TRANSITION_MAX_LINE 16384u
#define TKM_FLOAT_TRANSITION_MAX_MANIFEST_BYTES 4096u

static int tkm_json_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static const char* tkm_json_skip_ws(const char* text) {
    while (*text != '\0' && tkm_json_is_space(*text)) {
        text++;
    }
    return text;
}

static const char* tkm_json_find_value(const char* text, const char* key) {
    char pattern[96];
    const char* found;
    const char* colon;

    if (!text || !key) {
        return 0;
    }

    if (snprintf(pattern, sizeof(pattern), "\"%s\"", key) >= (int)sizeof(pattern)) {
        return 0;
    }

    found = strstr(text, pattern);
    if (!found) {
        return 0;
    }

    colon = strchr(found + strlen(pattern), ':');
    if (!colon) {
        return 0;
    }

    return tkm_json_skip_ws(colon + 1);
}

static int tkm_json_parse_string(const char* text, const char* key, char* out, uint32_t out_count) {
    const char* value = tkm_json_find_value(text, key);
    uint32_t i = 0;

    if (!value || !out || out_count == 0 || *value != '"') {
        return TKM_ERR;
    }

    value++;
    while (value[i] != '\0' && value[i] != '"') {
        if (value[i] == '\\' || i + 1 >= out_count) {
            return TKM_ERR;
        }
        out[i] = value[i];
        i++;
    }

    if (value[i] != '"') {
        return TKM_ERR;
    }

    out[i] = '\0';
    return TKM_OK;
}

static int tkm_json_parse_exporter_metadata(const char* text, char* out, uint32_t out_count) {
    const char* value;

    if (!text || !out || out_count == 0) {
        return TKM_ERR;
    }

    if (tkm_json_parse_string(text, "exporter", out, out_count) == TKM_OK) {
        return TKM_OK;
    }

    value = tkm_json_find_value(text, "exporter");
    if (!value) {
        return TKM_ERR;
    }
    value = tkm_json_skip_ws(value);
    if (*value != '{' || snprintf(out, out_count, "metadata") >= (int)out_count) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int tkm_json_parse_i32(const char* text, const char* key, int32_t* out) {
    const char* value = tkm_json_find_value(text, key);
    char* end = 0;
    long parsed;

    if (!value || !out) {
        return TKM_ERR;
    }

    parsed = strtol(value, &end, 10);
    if (end == value || parsed < INT32_MIN || parsed > INT32_MAX) {
        return TKM_ERR;
    }

    *out = (int32_t)parsed;
    return TKM_OK;
}

static int tkm_json_parse_u32(const char* text, const char* key, uint32_t* out) {
    int32_t parsed = 0;

    if (!out || tkm_json_parse_i32(text, key, &parsed) != TKM_OK || parsed < 0) {
        return TKM_ERR;
    }

    *out = (uint32_t)parsed;
    return TKM_OK;
}

static int tkm_json_parse_f32(const char* text, const char* key, float* out) {
    const char* value = tkm_json_find_value(text, key);
    char* end = 0;
    float parsed;

    if (!value || !out) {
        return TKM_ERR;
    }

    parsed = strtof(value, &end);
    if (end == value) {
        return TKM_ERR;
    }

    *out = parsed;
    return TKM_OK;
}

static int tkm_json_parse_obs(const char* text, float* obs, uint32_t expected_count) {
    const char* cursor = tkm_json_find_value(text, "obs");
    uint32_t count = 0;

    if (!cursor || !obs || expected_count > TKM_FLOAT_TRANSITION_MAX_OBS) {
        return TKM_ERR;
    }

    cursor = tkm_json_skip_ws(cursor);
    if (*cursor != '[') {
        return TKM_ERR;
    }
    cursor++;

    while (1) {
        char* end = 0;
        float value;

        cursor = tkm_json_skip_ws(cursor);
        if (*cursor == ']') {
            cursor++;
            break;
        }

        value = strtof(cursor, &end);
        if (end == cursor || count >= expected_count) {
            return TKM_ERR;
        }

        obs[count++] = value;
        cursor = tkm_json_skip_ws(end);
        if (*cursor == ',') {
            cursor++;
        } else if (*cursor != ']') {
            return TKM_ERR;
        }
    }

    if (count != expected_count) {
        return TKM_ERR;
    }

    return TKM_OK;
}

void tkm_float_transition_dataset_init(TkmFloatTransitionDataset* dataset) {
    if (!dataset) {
        return;
    }

    dataset->row_count = 0;
    dataset->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
    dataset->action_count = TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
}

static int tkm_float_transition_manifest_validate(const TkmFloatTransitionManifest* manifest) {
    if (!manifest ||
        strcmp(manifest->env, "pufferlib_breakout") != 0 ||
        manifest->source[0] == '\0' ||
        manifest->obs_dim != TKM_PUFFERLIB_BREAKOUT_OBS_DIM ||
        manifest->action_count != TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        return TKM_ERR;
    }

    return TKM_OK;
}

int tkm_float_transition_manifest_load(TkmFloatTransitionManifest* manifest, const char* path) {
    FILE* file;
    char text[TKM_FLOAT_TRANSITION_MAX_MANIFEST_BYTES + 1u];
    size_t bytes_read;

    if (!manifest || !path) {
        return TKM_ERR;
    }

    file = fopen(path, "rb");
    if (!file) {
        return TKM_ERR;
    }

    bytes_read = fread(text, 1u, TKM_FLOAT_TRANSITION_MAX_MANIFEST_BYTES, file);
    if (ferror(file) || (!feof(file) && bytes_read == TKM_FLOAT_TRANSITION_MAX_MANIFEST_BYTES)) {
        fclose(file);
        return TKM_ERR;
    }
    if (fclose(file) != 0) {
        return TKM_ERR;
    }
    text[bytes_read] = '\0';

    memset(manifest, 0, sizeof(*manifest));
    if (tkm_json_parse_string(text, "env", manifest->env, TKM_FLOAT_TRANSITION_MAX_TEXT) != TKM_OK ||
        tkm_json_parse_string(text, "source", manifest->source, TKM_FLOAT_TRANSITION_MAX_TEXT) != TKM_OK ||
        tkm_json_parse_exporter_metadata(text, manifest->exporter, TKM_FLOAT_TRANSITION_MAX_TEXT) != TKM_OK ||
        tkm_json_parse_u32(text, "obs_dim", &manifest->obs_dim) != TKM_OK ||
        tkm_json_parse_u32(text, "action_count", &manifest->action_count) != TKM_OK ||
        tkm_json_parse_u32(text, "row_count", &manifest->row_count) != TKM_OK ||
        tkm_json_parse_u32(text, "episode_count", &manifest->episode_count) != TKM_OK ||
        tkm_json_parse_f32(text, "min_return", &manifest->min_return) != TKM_OK ||
        tkm_json_parse_f32(text, "max_return", &manifest->max_return) != TKM_OK ||
        tkm_json_parse_f32(text, "mean_return", &manifest->mean_return) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_float_transition_manifest_validate(manifest);
}

static int tkm_float_transition_parse_row(TkmFloatTransition* row, const char* line) {
    uint32_t action = 0;
    uint32_t terminal = 0;

    if (!row || !line) {
        return TKM_ERR;
    }

    memset(row, 0, sizeof(*row));
    if (tkm_json_parse_u32(line, "version", &row->version) != TKM_OK ||
        tkm_json_parse_string(line, "env", row->env, TKM_FLOAT_TRANSITION_MAX_TEXT) != TKM_OK ||
        tkm_json_parse_string(line, "source", row->source, TKM_FLOAT_TRANSITION_MAX_TEXT) != TKM_OK ||
        tkm_json_parse_i32(line, "env_index", &row->env_index) != TKM_OK ||
        tkm_json_parse_i32(line, "episode", &row->episode) != TKM_OK ||
        tkm_json_parse_i32(line, "t", &row->t) != TKM_OK ||
        tkm_json_parse_u32(line, "obs_dim", &row->obs_dim) != TKM_OK ||
        tkm_json_parse_obs(line, row->obs, TKM_PUFFERLIB_BREAKOUT_OBS_DIM) != TKM_OK ||
        tkm_json_parse_u32(line, "action", &action) != TKM_OK ||
        tkm_json_parse_f32(line, "reward", &row->reward) != TKM_OK ||
        tkm_json_parse_u32(line, "terminal", &terminal) != TKM_OK) {
        return TKM_ERR;
    }

    if (row->version != 1u ||
        strcmp(row->env, "pufferlib_breakout") != 0 ||
        row->source[0] == '\0' ||
        row->env_index < 0 ||
        row->episode < 0 ||
        row->t < 0 ||
        row->obs_dim != TKM_PUFFERLIB_BREAKOUT_OBS_DIM ||
        action >= TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT ||
        terminal > 1u) {
        return TKM_ERR;
    }

    row->action = (uint8_t)action;
    row->terminal = (uint8_t)terminal;
    return TKM_OK;
}

int tkm_float_transition_dataset_load_jsonl(TkmFloatTransitionDataset* dataset, const char* path) {
    FILE* file;
    char line[TKM_FLOAT_TRANSITION_MAX_LINE];

    if (!dataset || !path) {
        return TKM_ERR;
    }

    file = fopen(path, "rb");
    if (!file) {
        return TKM_ERR;
    }

    tkm_float_transition_dataset_init(dataset);
    while (fgets(line, sizeof(line), file)) {
        size_t len = strlen(line);
        TkmFloatTransition row;

        if (len + 1u == sizeof(line) && line[len - 1u] != '\n') {
            fclose(file);
            return TKM_ERR;
        }
        if (dataset->row_count >= TKM_FLOAT_TRANSITION_MAX_ROWS) {
            fclose(file);
            return TKM_ERR;
        }
        if (tkm_float_transition_parse_row(&row, line) != TKM_OK) {
            fclose(file);
            return TKM_ERR;
        }

        dataset->rows[dataset->row_count++] = row;
    }

    if (ferror(file)) {
        fclose(file);
        return TKM_ERR;
    }
    if (fclose(file) != 0 || dataset->row_count == 0) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int tkm_float_episode_matches(const TkmFloatEpisodeSummary* episode, const TkmFloatTransition* row) {
    return episode->env_index == row->env_index && episode->episode == row->episode;
}

static int tkm_float_episode_find(
    const TkmFloatEpisodeSummary* episodes,
    uint32_t episode_count,
    const TkmFloatTransition* row,
    uint32_t* out_index
) {
    if (!episodes || !row || !out_index) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < episode_count; i++) {
        if (tkm_float_episode_matches(&episodes[i], row)) {
            *out_index = i;
            return TKM_OK;
        }
    }

    return TKM_ERR;
}

int tkm_float_transition_episode_returns(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatEpisodeSummary* episodes,
    uint32_t max_episodes,
    uint32_t* out_episode_count
) {
    uint32_t episode_count = 0;

    if (!dataset || !episodes || !out_episode_count) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < dataset->row_count; i++) {
        const TkmFloatTransition* row = &dataset->rows[i];
        uint32_t episode_index = 0;
        if (tkm_float_episode_find(episodes, episode_count, row, &episode_index) != TKM_OK) {
            if (episode_count >= max_episodes) {
                return TKM_ERR;
            }
            episode_index = episode_count++;
            episodes[episode_index].env_index = row->env_index;
            episodes[episode_index].episode = row->episode;
            episodes[episode_index].row_count = 0;
            episodes[episode_index].episode_return = 0.0f;
        }

        episodes[episode_index].row_count++;
        episodes[episode_index].episode_return += row->reward;
    }

    *out_episode_count = episode_count;
    return TKM_OK;
}

static int tkm_float_transition_copy_selected(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    const TkmFloatEpisodeSummary* episodes,
    const uint8_t* selected,
    uint32_t episode_count
) {
    if (!dataset || !out || !episodes || !selected) {
        return TKM_ERR;
    }

    tkm_float_transition_dataset_init(out);
    for (uint32_t i = 0; i < dataset->row_count; i++) {
        const TkmFloatTransition* row = &dataset->rows[i];
        uint32_t episode_index = 0;
        if (tkm_float_episode_find(episodes, episode_count, row, &episode_index) != TKM_OK) {
            return TKM_ERR;
        }
        if (!selected[episode_index]) {
            continue;
        }
        if (out->row_count >= TKM_FLOAT_TRANSITION_MAX_ROWS) {
            return TKM_ERR;
        }
        out->rows[out->row_count++] = *row;
    }

    return TKM_OK;
}

int tkm_float_transition_filter_all(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out
) {
    if (!dataset || !out) {
        return TKM_ERR;
    }

    *out = *dataset;
    return TKM_OK;
}

int tkm_float_transition_filter_min_return(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    float min_return
) {
    TkmFloatEpisodeSummary episodes[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint8_t selected[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint32_t episode_count = 0;

    if (tkm_float_transition_episode_returns(dataset, episodes, TKM_FLOAT_TRANSITION_MAX_EPISODES, &episode_count) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < episode_count; i++) {
        selected[i] = episodes[i].episode_return >= min_return ? 1u : 0u;
    }

    return tkm_float_transition_copy_selected(dataset, out, episodes, selected, episode_count);
}

int tkm_float_transition_filter_top_return(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    uint32_t max_episodes
) {
    TkmFloatEpisodeSummary episodes[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint8_t selected[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint32_t episode_count = 0;
    uint32_t select_count;

    if (max_episodes == 0 ||
        tkm_float_transition_episode_returns(dataset, episodes, TKM_FLOAT_TRANSITION_MAX_EPISODES, &episode_count) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < episode_count; i++) {
        selected[i] = 0u;
    }

    select_count = max_episodes < episode_count ? max_episodes : episode_count;
    for (uint32_t pick = 0; pick < select_count; pick++) {
        uint32_t best = 0;
        uint8_t found = 0;
        for (uint32_t i = 0; i < episode_count; i++) {
            if (selected[i]) {
                continue;
            }
            if (!found || episodes[i].episode_return > episodes[best].episode_return) {
                best = i;
                found = 1u;
            }
        }
        if (!found) {
            return TKM_ERR;
        }
        selected[best] = 1u;
    }

    return tkm_float_transition_copy_selected(dataset, out, episodes, selected, episode_count);
}

int tkm_float_transition_filter_return_range(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    float min_return,
    float max_return
) {
    TkmFloatEpisodeSummary episodes[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint8_t selected[TKM_FLOAT_TRANSITION_MAX_EPISODES];
    uint32_t episode_count = 0;

    if (min_return > max_return ||
        tkm_float_transition_episode_returns(dataset, episodes, TKM_FLOAT_TRANSITION_MAX_EPISODES, &episode_count) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < episode_count; i++) {
        selected[i] = episodes[i].episode_return >= min_return && episodes[i].episode_return <= max_return ? 1u : 0u;
    }

    return tkm_float_transition_copy_selected(dataset, out, episodes, selected, episode_count);
}
