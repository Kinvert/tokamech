#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "breakout.h"

#include "core/dataset/float_transition.h"
#include "core/sequence/layout.h"
#include "projects/breakout/pufferlib_policy.h"

#ifndef PUFFERLIB_DIR
#define PUFFERLIB_DIR "/home/claude/pathfinder"
#endif

#define DEFAULT_TOKEN_CONFIG_PATH "projects/breakout/pufferlib_token_ngram.ini"
#define TOKEN_VIZ_TOPK 5u
#define TOKEN_VIZ_SIDE_PANEL_WIDTH 320

static uint32_t g_configured_frameskip = 4u;

typedef struct {
    uint16_t token_class;
    float score;
} TokenVizCandidate;

typedef struct {
    BreakoutPufferlibTokenPolicy* policy;
    const BreakoutPufferlibTokenReport* report;
    BreakoutPufferlibRenderVisualizationKind kind;
    uint8_t last_action;
    uint32_t frame;
} BreakoutTokenVizState;

static void fail_if(int status, const char* label) {
    if (status != TKM_OK) {
        fprintf(stderr, "%s failed\n", label);
        exit(1);
    }
}

static uint32_t read_u32_env(const char* name, uint32_t fallback) {
    const char* text = getenv(name);
    char* end = NULL;
    unsigned long value;

    if (!text || text[0] == '\0') {
        return fallback;
    }
    value = strtoul(text, &end, 10);
    if (end == text || value > 0xfffffffful) {
        return fallback;
    }
    return (uint32_t)value;
}

static int read_ini_u32(const TkmIni* ini, const char* section, const char* key, uint32_t fallback, uint32_t* out) {
    int32_t parsed = 0;

    if (!out) {
        return TKM_ERR;
    }
    if (tkm_ini_get_i32(ini, section, key, (int32_t)fallback, &parsed) != TKM_OK || parsed < 0) {
        return TKM_ERR;
    }
    *out = (uint32_t)parsed;
    return TKM_OK;
}

static int read_ini_render_visualization(
    const TkmIni* ini,
    BreakoutPufferlibRenderVisualizationKind* out
) {
    const char* text;

    if (!ini || !out) {
        return TKM_ERR;
    }
    text = tkm_ini_get(ini, "render", "token_visualization", "topk");
    return breakout_pufferlib_render_visualization_kind_from_string(text, out);
}

static int load_token_config(
    const char* path,
    TkmIni* ini,
    BreakoutPufferlibTokenConfig* config,
    const char** out_dataset_path,
    uint32_t* out_frameskip,
    BreakoutPufferlibRenderVisualizationKind* out_render_visualization
) {
    uint32_t frameskip = 4u;

    if (!path || !ini || !config || !out_dataset_path || !out_frameskip ||
        !out_render_visualization) {
        return TKM_ERR;
    }
    if (tkm_ini_parse_file(ini, path) != TKM_OK ||
        breakout_pufferlib_token_config_from_ini(ini, config) != TKM_OK ||
        read_ini_u32(ini, "breakout.pufferlib", "frameskip", 4u, &frameskip) != TKM_OK ||
        frameskip == 0u ||
        read_ini_render_visualization(ini, out_render_visualization) != TKM_OK) {
        return TKM_ERR;
    }
    *out_dataset_path = tkm_ini_get(ini, "breakout.pufferlib", "dataset", "");
    *out_frameskip = frameskip;
    return TKM_OK;
}

static Breakout make_pufferlib_breakout_env(void) {
    Breakout env = {
        .client = NULL,
        .num_agents = 1,
        .frameskip = (int)read_u32_env("TKM_PUFFERLIB_BREAKOUT_FRAMESKIP", g_configured_frameskip),
        .width = 576,
        .height = 330,
        .initial_paddle_width = 62,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .initial_ball_speed = 256,
        .max_ball_speed = 448,
        .paddle_speed = 620,
        .continuous = 0,
        .rng = 1u,
    };
    allocate(&env);
    c_reset(&env);
    return env;
}

static const char* token_mode_name(const BreakoutPufferlibTokenReport* report) {
    if (strcmp(report->sequence_model_name, "token_backoff_ngram") == 0) {
        return "token_backoff_ngram";
    }
    if (strcmp(report->sequence_model_name, "token_mlp_window") == 0) {
        return "token_mlp_window";
    }
    if (strcmp(report->sequence_model_name, "token_ngram") == 0) {
        return "token_ngram";
    }
    return "token_policy";
}

static uint8_t token_viz_enabled(
    BreakoutPufferlibRenderVisualizationKind selected,
    BreakoutPufferlibRenderVisualizationKind panel
) {
    return selected == panel || selected == BREAKOUT_PUFFERLIB_RENDER_VIS_ALL;
}

static const char* token_viz_type_name(uint8_t type) {
    if (type == TKM_SEQUENCE_PAD) {
        return "PAD";
    }
    if (type == TKM_SEQUENCE_BOS) {
        return "BOS";
    }
    if (type == TKM_SEQUENCE_OBS) {
        return "OBS";
    }
    if (type == TKM_SEQUENCE_ACTION) {
        return "ACT";
    }
    return "TOK";
}

static const char* token_viz_action_name(uint8_t action) {
    if (action == LEFT) {
        return "LEFT";
    }
    if (action == RIGHT) {
        return "RIGHT";
    }
    return "NOOP";
}

static Color token_viz_type_color(uint8_t type) {
    if (type == TKM_SEQUENCE_OBS) {
        return (Color){0, 188, 212, 235};
    }
    if (type == TKM_SEQUENCE_ACTION) {
        return (Color){255, 152, 0, 240};
    }
    if (type == TKM_SEQUENCE_BOS) {
        return (Color){156, 204, 101, 225};
    }
    return (Color){120, 144, 156, 210};
}

static uint8_t token_viz_quantize(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t dim,
    float value
) {
    float min_value = policy->obs_min[dim];
    float max_value = policy->obs_max[dim];
    float span = max_value - min_value;
    float scaled;
    int token;

    if (span <= 0.000001f || policy->bin_count <= 1u) {
        return 0u;
    }
    if (value < min_value) {
        value = min_value;
    }
    if (value > max_value) {
        value = max_value;
    }
    scaled = (value - min_value) / span;
    token = (int)(scaled * (float)policy->bin_count);
    if (token < 0) {
        token = 0;
    }
    if (token >= (int)policy->bin_count) {
        token = (int)policy->bin_count - 1;
    }
    return (uint8_t)token;
}

static uint32_t token_viz_hash_mix(uint32_t hash, uint32_t value) {
    hash ^= value + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
    return hash;
}

static uint32_t token_viz_context_key(const uint32_t* context, uint32_t context_count) {
    uint32_t hash = 2166136261u;

    for (uint32_t i = 0u; i < context_count; i++) {
        hash = token_viz_hash_mix(hash, context[i]);
    }
    return hash == 0u ? 1u : hash;
}

static const BreakoutPufferlibTokenNgramEntry* token_viz_ngram_entry(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t key
) {
    uint32_t start = key % BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES;

    for (uint32_t probe = 0u; probe < 32u; probe++) {
        uint32_t index = (start + probe) % BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES;
        const BreakoutPufferlibTokenNgramEntry* entry = &policy->ngram_entries[index];

        if (entry->used && entry->key == key) {
            return entry;
        }
        if (!entry->used) {
            return NULL;
        }
    }
    return NULL;
}

static void token_viz_class_label(
    const BreakoutPufferlibTokenPolicy* policy,
    uint16_t token_class,
    char* out,
    size_t out_count
) {
    uint32_t obs_class = token_class >= TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT ?
        (uint32_t)token_class - TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT : 0u;
    uint32_t selected_index = policy->bin_count > 0u ? obs_class / policy->bin_count : 0u;
    uint32_t bin = policy->bin_count > 0u ? obs_class % policy->bin_count : 0u;

    if (token_class < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        snprintf(out, out_count, "act:%s", token_viz_action_name((uint8_t)token_class));
        return;
    }
    if (selected_index < policy->selected_dim_count) {
        snprintf(out, out_count, "obs:d%u=%u", policy->selected_dims[selected_index], bin);
        return;
    }
    snprintf(out, out_count, "tok:%u", token_class);
}

static void token_viz_code_label(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t code,
    char* out,
    size_t out_count
) {
    uint8_t type = (uint8_t)((code >> 16u) & 0xffu);
    uint16_t token = (uint16_t)(code & 0xffffu);

    if (type == TKM_SEQUENCE_ACTION) {
        snprintf(out, out_count, "ACT %s", token_viz_action_name((uint8_t)token));
        return;
    }
    if (type == TKM_SEQUENCE_OBS && policy->bin_count > 0u) {
        uint32_t selected_index = token / policy->bin_count;
        uint32_t bin = token % policy->bin_count;
        if (selected_index < policy->selected_dim_count) {
            snprintf(out, out_count, "OBS d%u:%u", policy->selected_dims[selected_index], bin);
            return;
        }
    }
    snprintf(out, out_count, "%s %u", token_viz_type_name(type), token);
}

static void token_viz_insert_candidate(
    TokenVizCandidate* candidates,
    uint16_t token_class,
    float score
) {
    for (uint32_t i = 0u; i < TOKEN_VIZ_TOPK; i++) {
        if (score <= candidates[i].score) {
            continue;
        }
        for (uint32_t j = TOKEN_VIZ_TOPK - 1u; j > i; j--) {
            candidates[j] = candidates[j - 1u];
        }
        candidates[i].token_class = token_class;
        candidates[i].score = score;
        return;
    }
}

static void token_viz_fill_ngram_topk(
    const BreakoutPufferlibTokenPolicy* policy,
    TokenVizCandidate* candidates
) {
    uint32_t key = token_viz_context_key(policy->stream_context, policy->stream_context_count);
    const BreakoutPufferlibTokenNgramEntry* entry = token_viz_ngram_entry(policy, key);
    uint32_t output_dim = TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT +
        policy->selected_dim_count * policy->bin_count;

    if (entry) {
        for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
            if (entry->counts[i] > 0u) {
                token_viz_insert_candidate(candidates, entry->classes[i], (float)entry->counts[i]);
            }
        }
    }
    if (candidates[0].score > 0.0f) {
        return;
    }
    for (uint32_t token_class = 0u; token_class < output_dim && token_class < TKM_MLP_WINDOW_MAX_OUTPUT; token_class++) {
        if (policy->ngram_class_prior[token_class] > 0u) {
            token_viz_insert_candidate(
                candidates,
                (uint16_t)token_class,
                (float)policy->ngram_class_prior[token_class]
            );
        }
    }
}

static int token_viz_build_mlp_input(
    const BreakoutPufferlibTokenPolicy* policy,
    float* input,
    uint32_t input_capacity
) {
    uint32_t input_dim = policy->history * 2u;
    uint32_t context_count = policy->stream_context_count;
    uint32_t pad_count;
    float token_scale;
    uint32_t input_index = 0u;

    if (!policy || !input || input_capacity < input_dim || input_dim == 0u) {
        return TKM_ERR;
    }
    if (context_count > policy->history) {
        context_count = policy->history;
    }
    pad_count = policy->history - context_count;
    token_scale = (float)(policy->selected_dim_count * policy->bin_count);
    if (token_scale < (float)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        token_scale = (float)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
    }
    if (token_scale <= 0.0f) {
        return TKM_ERR;
    }
    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        uint8_t type = TKM_SEQUENCE_PAD;
        uint32_t token = 0u;

        if (slot >= pad_count) {
            uint32_t code = policy->stream_context[slot - pad_count];
            type = (uint8_t)((code >> 16u) & 0xffu);
            token = code & 0xffffu;
        }
        if (type > TKM_SEQUENCE_TRANSITION) {
            type = TKM_SEQUENCE_TRANSITION;
        }
        if ((float)token > token_scale) {
            token = (uint32_t)token_scale;
        }
        input[input_index++] = 2.0f * ((float)type / (float)TKM_SEQUENCE_TRANSITION) - 1.0f;
        input[input_index++] = 2.0f * ((float)token / token_scale) - 1.0f;
    }
    return input_index == input_dim ? TKM_OK : TKM_ERR;
}

static void token_viz_fill_mlp_topk(
    const BreakoutPufferlibTokenPolicy* policy,
    TokenVizCandidate* candidates
) {
    float input[TKM_MLP_WINDOW_MAX_INPUT];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t output_dim = TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT +
        policy->selected_dim_count * policy->bin_count;

    if (!policy->use_mlp ||
        output_dim == 0u ||
        output_dim > TKM_MLP_WINDOW_MAX_OUTPUT ||
        token_viz_build_mlp_input(policy, input, TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
        tkm_mlp_window_forward(&policy->mlp, input, logits, TKM_MLP_WINDOW_MAX_OUTPUT) != TKM_OK) {
        token_viz_insert_candidate(candidates, policy->stream_context_count > 0u ? 0u : 0u, 1.0f);
        return;
    }
    for (uint32_t token_class = 0u; token_class < output_dim; token_class++) {
        token_viz_insert_candidate(candidates, (uint16_t)token_class, logits[token_class]);
    }
}

static void token_viz_fill_topk(
    const BreakoutPufferlibTokenPolicy* policy,
    TokenVizCandidate* candidates
) {
    for (uint32_t i = 0u; i < TOKEN_VIZ_TOPK; i++) {
        candidates[i].token_class = 0u;
        candidates[i].score = -1000000000.0f;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
        token_viz_fill_mlp_topk(policy, candidates);
    } else {
        token_viz_fill_ngram_topk(policy, candidates);
    }
}

static void token_viz_draw_panel(int x, int y, int w, int h, const char* title) {
    DrawRectangle(x, y, w, h, (Color){4, 12, 18, 218});
    DrawRectangleLines(x, y, w, h, (Color){93, 230, 210, 220});
    DrawText(title, x + 8, y + 7, 12, (Color){220, 255, 250, 255});
}

static void token_viz_draw_topk(
    const BreakoutPufferlibTokenPolicy* policy,
    int x,
    int y,
    int w,
    int h
) {
    TokenVizCandidate candidates[TOKEN_VIZ_TOPK];
    float max_score;

    token_viz_draw_panel(x, y, w, h, "next-token scores");
    token_viz_fill_topk(policy, candidates);
    max_score = candidates[0].score;
    if (max_score <= 0.00001f) {
        max_score = 1.0f;
    }
    for (uint32_t i = 0u; i < TOKEN_VIZ_TOPK; i++) {
        char label[48];
        int row_y = y + 28 + (int)i * 19;
        int bar_w = (int)(168.0f * (candidates[i].score / max_score));
        Color color = candidates[i].token_class < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT ?
            (Color){255, 171, 64, 235} : (Color){77, 208, 225, 235};

        if (bar_w < 2) {
            bar_w = 2;
        }
        token_viz_class_label(policy, candidates[i].token_class, label, sizeof(label));
        DrawText(label, x + 8, row_y, 10, WHITE);
        DrawRectangle(x + w - 176, row_y + 1, bar_w, 11, color);
    }
}

static void token_viz_draw_token_stream(
    const BreakoutPufferlibTokenPolicy* policy,
    int x,
    int y,
    int w,
    int h
) {
    TokenVizCandidate candidates[TOKEN_VIZ_TOPK];
    uint32_t context_count = policy->stream_context_count;
    uint32_t pad_count;
    int row_h = 22;
    int history_y = y + 54;
    int pred_y = y + h - 84;
    char label[64];

    if (context_count > policy->history) {
        context_count = policy->history;
    }
    pad_count = policy->history - context_count;

    token_viz_draw_panel(x, y, w, h, "token-stream predictor");
    DrawText("CONTEXT WINDOW", x + 12, y + 28, 12, (Color){255, 245, 157, 255});
    DrawText("previous obs/action tokens", x + 126, y + 30, 10, (Color){176, 224, 230, 255});

    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        uint32_t code = 0u;
        uint8_t type = TKM_SEQUENCE_PAD;
        int row_y = history_y + (int)slot * row_h;
        int age = (int)(policy->history - 1u - slot);
        Color color;

        if (slot >= pad_count) {
            code = policy->stream_context[slot - pad_count];
            type = (uint8_t)((code >> 16u) & 0xffu);
        }
        color = token_viz_type_color(type);
        token_viz_code_label(policy, code, label, sizeof(label));
        DrawRectangle(x + 12, row_y, w - 24, row_h - 4, (Color){12, 28, 36, 235});
        DrawRectangle(x + 76, row_y + 3, 44, row_h - 10, color);
        DrawText(TextFormat("t-%d", age), x + 20, row_y + 5, 10, (Color){190, 210, 215, 255});
        DrawText(token_viz_type_name(type), x + 84, row_y + 5, 10, BLACK);
        DrawText(label, x + 132, row_y + 5, 10, WHITE);
    }

    DrawLine(x + w / 2, pred_y - 29, x + w / 2, pred_y - 8, (Color){255, 213, 79, 255});
    DrawTriangle(
        (Vector2){(float)(x + w / 2 - 5), (float)(pred_y - 9)},
        (Vector2){(float)(x + w / 2 + 5), (float)(pred_y - 9)},
        (Vector2){(float)(x + w / 2), (float)(pred_y - 1)},
        (Color){255, 213, 79, 255});

    token_viz_fill_topk(policy, candidates);
    token_viz_class_label(policy, candidates[0].token_class, label, sizeof(label));
    DrawRectangle(x + 12, pred_y, w - 24, 58, (Color){48, 33, 8, 245});
    DrawRectangleLinesEx((Rectangle){(float)(x + 12), (float)pred_y, (float)(w - 24), 58.0f},
        3.0f,
        (Color){255, 213, 79, 255});
    DrawText("PREDICTED NEXT TOKEN", x + 24, pred_y + 8, 12, (Color){255, 245, 157, 255});
    DrawText(label, x + 24, pred_y + 27, 18, WHITE);
    DrawText("same stream: obs bins + actions", x + 24, pred_y + 48, 10, (Color){255, 224, 130, 255});

    DrawText("top alternatives", x + 12, pred_y + 68, 10, (Color){176, 224, 230, 255});
    for (uint32_t i = 1u; i < TOKEN_VIZ_TOPK && i < 4u; i++) {
        int alt_y = pred_y + 82 + (int)(i - 1u) * 14;
        token_viz_class_label(policy, candidates[i].token_class, label, sizeof(label));
        DrawText(TextFormat("%u. %s", i + 1u, label), x + 22, alt_y, 10, (Color){220, 235, 238, 255});
    }

    DrawText("model generates one categorical token at a time",
        x + 12,
        y + h - 14,
        10,
        (Color){158, 255, 240, 255});
}

static int token_viz_build_continuous_mlp_input(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    float* input,
    uint32_t input_capacity
) {
    uint32_t input_dim;
    uint32_t history_count;
    uint32_t pad_count;
    uint32_t input_index = 0u;

    if (!policy || !obs || !input ||
        policy->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW ||
        policy->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW ||
        policy->selected_dim_count == 0u ||
        policy->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    input_dim = policy->token_mlp_input_dim;
    if (input_dim == 0u || input_capacity < input_dim) {
        return TKM_ERR;
    }
    history_count = policy->value_history_count;
    if (history_count > policy->history) {
        history_count = policy->history;
    }
    pad_count = policy->history - history_count;
    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            if (slot < pad_count) {
                input[input_index++] = 0.0f;
            } else {
                input[input_index++] =
                    policy->value_history[
                        (slot - pad_count) * BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS + i];
            }
        }
    }
    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        uint32_t dim = policy->selected_dims[i];
        if (dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        input[input_index++] = obs[dim];
    }
    return input_index == input_dim ? TKM_OK : TKM_ERR;
}

static void token_viz_fill_continuous_mlp_actions(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    TokenVizCandidate* candidates
) {
    float input[TKM_MLP_WINDOW_MAX_INPUT];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];

    for (uint32_t i = 0u; i < TOKEN_VIZ_TOPK; i++) {
        candidates[i].token_class = 0u;
        candidates[i].score = -1000000000.0f;
    }
    if (!policy->use_mlp ||
        token_viz_build_continuous_mlp_input(policy, obs, input, TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
        tkm_mlp_window_forward(&policy->mlp, input, logits, TKM_MLP_WINDOW_MAX_OUTPUT) != TKM_OK) {
        token_viz_insert_candidate(candidates, 0u, 1.0f);
        return;
    }
    for (uint32_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        token_viz_insert_candidate(candidates, (uint16_t)action, logits[action]);
    }
}

static Color token_viz_continuous_value_color(float value) {
    float pct = (value + 1.0f) * 0.5f;
    unsigned char r;
    unsigned char g;
    unsigned char b;

    if (pct < 0.0f) {
        pct = 0.0f;
    }
    if (pct > 1.0f) {
        pct = 1.0f;
    }
    r = (unsigned char)(30.0f + 220.0f * pct);
    g = (unsigned char)(90.0f + 120.0f * (1.0f - pct));
    b = (unsigned char)(220.0f - 150.0f * pct);
    return (Color){r, g, b, 235};
}

static void token_viz_draw_continuous_mlp(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    int x,
    int y,
    int w,
    int h
) {
    TokenVizCandidate candidates[TOKEN_VIZ_TOPK];
    uint32_t history_count = policy->value_history_count;
    uint32_t pad_count;
    int row_h = 18;
    int history_y = y + 54;
    int current_y = y + 206;
    int pred_y = y + h - 76;
    float best_score;

    if (history_count > policy->history) {
        history_count = policy->history;
    }
    pad_count = policy->history - history_count;

    token_viz_draw_panel(x, y, w, h, "continuous MLP policy");
    DrawText("VALUE WINDOW", x + 12, y + 28, 12, (Color){255, 245, 157, 255});
    DrawText("8 history rows x 16 selected obs dims", x + 116, y + 30, 10, (Color){176, 224, 230, 255});

    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        int row_y = history_y + (int)slot * row_h;
        int age = (int)(policy->history - 1u - slot);

        DrawText(TextFormat("t-%d", age), x + 12, row_y + 4, 10, (Color){190, 210, 215, 255});
        for (uint32_t i = 0u; i < policy->selected_dim_count && i < 16u; i++) {
            float value = 0.0f;
            int cell_x = x + 48 + (int)i * 14;
            if (slot >= pad_count) {
                value = policy->value_history[
                    (slot - pad_count) * BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS + i];
            }
            DrawRectangle(cell_x, row_y + 3, 11, 11, token_viz_continuous_value_color(value));
        }
    }

    DrawText("CURRENT OBS", x + 12, current_y - 14, 12, (Color){255, 245, 157, 255});
    for (uint32_t i = 0u; i < policy->selected_dim_count && i < 16u; i++) {
        uint32_t dim = policy->selected_dims[i];
        int cell_x = x + 48 + (int)i * 14;
        float value = dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM ? obs[dim] : 0.0f;
        DrawRectangle(cell_x, current_y, 11, 15, token_viz_continuous_value_color(value));
    }
    DrawText("selected dims: 0..15", x + 12, current_y + 22, 10, (Color){176, 224, 230, 255});
    DrawText("MLP: 144 -> 64 -> 3 actions", x + 12, current_y + 38, 10, (Color){176, 224, 230, 255});

    token_viz_fill_continuous_mlp_actions(policy, obs, candidates);
    best_score = candidates[0].score;
    if (best_score < 0.00001f) {
        best_score = 1.0f;
    }
    DrawRectangle(x + 12, pred_y, w - 24, 58, (Color){48, 33, 8, 245});
    DrawRectangleLinesEx((Rectangle){(float)(x + 12), (float)pred_y, (float)(w - 24), 58.0f},
        3.0f,
        (Color){255, 213, 79, 255});
    DrawText("PREDICTED ACTION", x + 24, pred_y + 8, 12, (Color){255, 245, 157, 255});
    DrawText(token_viz_action_name((uint8_t)candidates[0].token_class), x + 24, pred_y + 27, 18, WHITE);

    for (uint32_t i = 0u; i < 3u; i++) {
        int bar_y = pred_y + 68 + (int)i * 15;
        int bar_w = (int)(120.0f * (candidates[i].score / best_score));
        if (bar_w < 2) {
            bar_w = 2;
        }
        DrawText(token_viz_action_name((uint8_t)candidates[i].token_class), x + 16, bar_y, 10, WHITE);
        DrawRectangle(x + 72, bar_y + 2, bar_w, 8, (Color){255, 171, 64, 235});
    }
}

static void token_viz_draw_context_grid(
    const BreakoutPufferlibTokenPolicy* policy,
    int x,
    int y,
    int w,
    int h
) {
    int cell_w = (w - 18) / 4;

    token_viz_draw_panel(x, y, w, h, "mlp context grid");
    for (uint32_t i = 0u; i < policy->history; i++) {
        uint32_t row = i / 4u;
        uint32_t col = i % 4u;
        uint32_t code = i < policy->stream_context_count ? policy->stream_context[i] : 0u;
        uint8_t type = (uint8_t)((code >> 16u) & 0xffu);
        char label[32];
        int cx = x + 8 + (int)col * cell_w;
        int cy = y + 28 + (int)row * 34;

        token_viz_code_label(policy, code, label, sizeof(label));
        DrawRectangle(cx, cy, cell_w - 5, 28, token_viz_type_color(type));
        DrawText(label, cx + 4, cy + 9, 10, BLACK);
    }
    DrawText("fixed window -> categorical next token", x + 8, y + h - 17, 10, (Color){200, 230, 230, 255});
}

static void token_viz_draw_quantization(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    int x,
    int y,
    int w,
    int h
) {
    uint32_t rows = policy->selected_dim_count < 6u ? policy->selected_dim_count : 6u;

    token_viz_draw_panel(x, y, w, h, "raw obs -> tokens");
    for (uint32_t i = 0u; i < rows; i++) {
        uint32_t dim = policy->selected_dims[i];
        uint8_t bin = token_viz_quantize(policy, dim, obs[dim]);
        int row_y = y + 27 + (int)i * 18;
        float min_value = policy->obs_min[dim];
        float max_value = policy->obs_max[dim];
        float span = max_value - min_value;
        float pct = span > 0.000001f ? (obs[dim] - min_value) / span : 0.0f;
        int dot_x;

        if (pct < 0.0f) {
            pct = 0.0f;
        }
        if (pct > 1.0f) {
            pct = 1.0f;
        }
        dot_x = x + 92 + (int)(pct * 92.0f);
        DrawText(TextFormat("d%u %.2f", dim, obs[dim]), x + 8, row_y, 10, WHITE);
        DrawRectangle(x + 92, row_y + 4, 92, 4, (Color){60, 82, 92, 255});
        DrawCircle(dot_x, row_y + 6, 4.0f, (Color){255, 235, 59, 255});
        DrawText(TextFormat("bin %u", bin), x + w - 46, row_y, 10, (Color){158, 255, 240, 255});
    }
}

static void token_viz_draw_pipeline(
    const BreakoutTokenVizState* viz,
    int x,
    int y,
    int w,
    int h
) {
    const char* steps[] = {
        "observe",
        "quantize",
        "append obs",
        "predict token",
        "execute action",
        "append action"
    };
    uint32_t active = viz->frame % 6u;

    token_viz_draw_panel(x, y, w, h, "autoregressive loop");
    for (uint32_t i = 0u; i < 6u; i++) {
        int row_y = y + 27 + (int)i * 18;
        Color color = i == active ? (Color){255, 213, 79, 255} : (Color){120, 144, 156, 255};
        DrawCircle(x + 15, row_y + 5, 5.0f, color);
        DrawText(steps[i], x + 27, row_y, 10, WHITE);
    }
    DrawText(TextFormat("last action: %s", token_viz_action_name(viz->last_action)),
        x + 8, y + h - 17, 10, (Color){255, 213, 79, 255});
}

static void token_viz_draw_transformer(
    const BreakoutPufferlibTokenPolicy* policy,
    int x,
    int y,
    int w,
    int h
) {
    token_viz_draw_panel(x, y, w, h, "transformer view");
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
        DrawText("No attention heads here.", x + 8, y + 30, 10, WHITE);
        DrawText("This model is a fixed-window MLP", x + 8, y + 47, 10, WHITE);
        DrawText("over token ids.", x + 8, y + 64, 10, WHITE);
    } else {
        DrawText("N-gram token context lookup.", x + 8, y + 30, 10, WHITE);
        DrawText("Future transformer: draw heads", x + 8, y + 47, 10, WHITE);
        DrawText("from token attention weights.", x + 8, y + 64, 10, WHITE);
    }
}

static void draw_breakout_token_viz_overlay(Breakout* env, void* user) {
    BreakoutTokenVizState* viz = (BreakoutTokenVizState*)user;
    BreakoutPufferlibTokenPolicy* policy;
    int right_x;
    int side_x;

    if (!env || !viz || !viz->policy || viz->kind == BREAKOUT_PUFFERLIB_RENDER_VIS_NONE) {
        return;
    }
    policy = viz->policy;
    right_x = env->width - 236;
    side_x = env->width + 12;

    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_TOPK)) {
        token_viz_draw_topk(policy, right_x, 34, 228, 126);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_TOKEN_STREAM)) {
        token_viz_draw_token_stream(policy, side_x, 12, TOKEN_VIZ_SIDE_PANEL_WIDTH - 24, env->height - 24);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_CONTINUOUS_MLP)) {
        token_viz_draw_continuous_mlp(
            policy,
            env->observations,
            side_x,
            12,
            TOKEN_VIZ_SIDE_PANEL_WIDTH - 24,
            env->height - 24);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_CONTEXT_GRID)) {
        token_viz_draw_context_grid(policy, right_x, 164, 228, 104);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_QUANTIZATION)) {
        token_viz_draw_quantization(policy, env->observations, right_x, 272, 228, 132);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_PIPELINE)) {
        token_viz_draw_pipeline(viz, right_x, 408, 228, 136);
    }
    if (token_viz_enabled(viz->kind, BREAKOUT_PUFFERLIB_RENDER_VIS_TRANSFORMER)) {
        token_viz_draw_transformer(policy, 8, 34, 220, 92);
    }
    DrawText(TextFormat("token predictor: %s", token_mode_name(viz->report)),
        8, env->height - 98, 12, (Color){220, 255, 250, 255});
}

int main(void) {
    const char* token_config_path = getenv("TKM_PUFFERLIB_BREAKOUT_CONFIG");
    const char* dataset_path = getenv("TKM_PUFFERLIB_BREAKOUT_JSONL");
    const char* config_dataset_path = "";
    const char* max_steps_text = getenv("TKM_PUFFERLIB_RENDER_MAX_STEPS");
    BreakoutPufferlibTokenPolicy token_policy;
    BreakoutPufferlibTokenConfig token_config;
    BreakoutPufferlibTokenReport token_report;
    BreakoutPufferlibRenderVisualizationKind render_visualization;
    BreakoutTokenVizState token_viz;
    TkmIni token_ini;
    static TkmFloatTransitionDataset dataset;
    Breakout env;
    uint32_t action_histogram[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT] = {0u, 0u, 0u};
    uint32_t steps = 0;
    uint32_t max_steps = 0;
    uint8_t action = NOOP;

    if (max_steps_text && max_steps_text[0] != '\0') {
        max_steps = (uint32_t)strtoul(max_steps_text, NULL, 10);
    }
    if (!token_config_path || token_config_path[0] == '\0') {
        token_config_path = DEFAULT_TOKEN_CONFIG_PATH;
    }
    fail_if(load_token_config(
            token_config_path,
            &token_ini,
            &token_config,
            &config_dataset_path,
            &g_configured_frameskip,
            &render_visualization),
        "load_token_config");
    if (!dataset_path || dataset_path[0] == '\0') {
        dataset_path = config_dataset_path;
    }
    if (!dataset_path || dataset_path[0] == '\0') {
        dataset_path = "/tmp/tkm_pufferlib_breakout_smoke/transitions-000000.jsonl";
    }

    fail_if(tkm_float_transition_dataset_load_jsonl(&dataset, dataset_path),
        "tkm_float_transition_dataset_load_jsonl");
    fail_if(breakout_pufferlib_token_policy_train(
            &token_policy, &dataset, &token_config, &token_report),
        "breakout_pufferlib_token_policy_train");
    printf("pufferlib breakout token_stack config=%s dataset=%s frameskip=%u\n",
        token_config_path,
        dataset_path,
        g_configured_frameskip);
    printf("pufferlib breakout token_layers layout=%s tokenizer=%s selection=%s input=%s model=%s head=%s\n",
        token_report.sequence_layout_name,
        token_report.observation_tokenizer_name,
        token_report.observation_selection_name,
        token_report.input_feature_name,
        token_report.sequence_model_name,
        token_report.action_head_name);
    printf("pufferlib breakout token_train rows=%u heldout_accuracy=%.3f obs_token_accuracy=%.3f action_token_accuracy=%.3f exact=%u backoff=%u prior=%u bins=%u dims=%u history=%u epochs=%u learning_rate=%.5f input_dim=%u hidden=%u\n",
        token_report.source_rows,
        token_report.heldout_accuracy,
        token_report.obs_token_accuracy,
        token_report.action_token_accuracy,
        token_report.exact_predictions,
        token_report.backoff_predictions,
        token_report.prior_predictions,
        token_report.bin_count,
        token_report.selected_dim_count,
        token_policy.history,
        token_config.epochs,
        token_config.learning_rate,
        token_policy.token_mlp_input_dim,
        token_policy.token_mlp_hidden_dim);

    if (chdir(PUFFERLIB_DIR) != 0) {
        perror("chdir PUFFERLIB_DIR");
        return 1;
    }

    env = make_pufferlib_breakout_env();
    breakout_set_render_overlay_panel_width(
        (token_viz_enabled(render_visualization, BREAKOUT_PUFFERLIB_RENDER_VIS_TOKEN_STREAM) ||
            token_viz_enabled(render_visualization, BREAKOUT_PUFFERLIB_RENDER_VIS_CONTINUOUS_MLP)) ?
            TOKEN_VIZ_SIDE_PANEL_WIDTH : 0);
    env.client = make_client(&env);
    token_viz.policy = &token_policy;
    token_viz.report = &token_report;
    token_viz.kind = render_visualization;
    token_viz.last_action = action;
    token_viz.frame = 0u;
    breakout_set_render_overlay(draw_breakout_token_viz_overlay, &token_viz);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            action = NOOP;
            breakout_pufferlib_token_policy_reset(&token_policy);
        }

        if (env.balls_fired == 0 && env.frameskip == 1) {
            action = NOOP;
        } else {
            fail_if(breakout_pufferlib_token_policy_predict(
                    &token_policy, env.observations, &action),
                "breakout_pufferlib_token_policy_predict");
        }

        env.actions[0] = (float)action;
        token_viz.last_action = action;
        token_viz.frame = steps;
        if (action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
            action_histogram[action]++;
        }
        c_step(&env);
        c_render(&env);
        steps++;
        if (env.terminals[0] != 0.0f) {
            breakout_pufferlib_token_policy_reset(&token_policy);
        }

        if (max_steps > 0u && steps >= max_steps) {
            break;
        }
    }

    close_client(env.client);
    free_allocated(&env);
    printf("pufferlib breakout render mode=%s steps=%u action_histogram=[%u,%u,%u]\n",
        token_mode_name(&token_report),
        steps,
        action_histogram[0],
        action_histogram[1],
        action_histogram[2]
    );
    return 0;
}
