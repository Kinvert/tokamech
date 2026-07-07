#include "projects/snake/snake.h"

#include <string.h>

typedef struct {
    int r;
    int c;
} SnakeCell;

static uint8_t snake_first_safe_action(const SnakeEnv* env);
static uint8_t snake_action_terminals(const SnakeEnv* env, uint8_t action);
static uint8_t snake_action_moves_into_old_tail(const SnakeEnv* env, uint8_t action);
static uint32_t snake_reachable_count(const SnakeEnv* env, int start_r, int start_c);
static uint8_t snake_action_keeps_space(const SnakeEnv* env, uint8_t action, uint32_t extra_space);
static int snake_food_distance_after(const SnakeEnv* env);
static int snake_score_action(const SnakeEnv* env, uint8_t action);

static int snake_index(const SnakeEnv* env, int r, int c) {
    return r * env->width + c;
}

static int snake_abs_int(int value) {
    return value < 0 ? -value : value;
}

static int snake_dr(uint8_t action) {
    if (action == SNAKE_UP) {
        return -1;
    }
    if (action == SNAKE_DOWN) {
        return 1;
    }
    return 0;
}

static int snake_dc(uint8_t action) {
    if (action == SNAKE_LEFT) {
        return -1;
    }
    if (action == SNAKE_RIGHT) {
        return 1;
    }
    return 0;
}

static uint8_t snake_left_of(uint8_t dir) {
    switch (dir) {
        case SNAKE_UP:
            return SNAKE_LEFT;
        case SNAKE_DOWN:
            return SNAKE_RIGHT;
        case SNAKE_LEFT:
            return SNAKE_DOWN;
        default:
            return SNAKE_UP;
    }
}

static uint8_t snake_right_of(uint8_t dir) {
    switch (dir) {
        case SNAKE_UP:
            return SNAKE_RIGHT;
        case SNAKE_DOWN:
            return SNAKE_LEFT;
        case SNAKE_LEFT:
            return SNAKE_UP;
        default:
            return SNAKE_DOWN;
    }
}

void snake_env_init_default(SnakeEnv* env) {
    memset(env, 0, sizeof(*env));
    env->width = SNAKE_WIDTH;
    env->height = SNAKE_HEIGHT;
    env->actions[0] = SNAKE_UP;
    env->actions[1] = SNAKE_DOWN;
    env->actions[2] = SNAKE_LEFT;
    env->actions[3] = SNAKE_RIGHT;
}

void snake_rebuild_grid(SnakeEnv* env) {
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            uint8_t tile = SNAKE_EMPTY;
            if (r == 0 || c == 0 || r == env->height - 1 || c == env->width - 1) {
                tile = SNAKE_WALL;
            }
            env->grid[snake_index(env, r, c)] = tile;
        }
    }

    for (int i = 0; i < env->length; i++) {
        env->grid[snake_index(env, env->body_r[i], env->body_c[i])] = SNAKE_BODY;
    }

    if (!env->terminal) {
        env->grid[snake_index(env, env->food_r, env->food_c)] = SNAKE_FOOD;
    }
}

void snake_compute_observations(SnakeEnv* env) {
    env->observations[0] = (int16_t)env->head_r;
    env->observations[1] = (int16_t)env->head_c;
    env->observations[2] = (int16_t)env->food_r;
    env->observations[3] = (int16_t)env->food_c;
    env->observations[4] = (int16_t)env->dir;
    env->observations[5] = (int16_t)env->grid[snake_index(env, env->head_r - 1, env->head_c)];
    env->observations[6] = (int16_t)env->grid[snake_index(env, env->head_r + 1, env->head_c)];
    env->observations[7] = (int16_t)env->grid[snake_index(env, env->head_r, env->head_c - 1)];
    env->observations[8] = (int16_t)env->grid[snake_index(env, env->head_r, env->head_c + 1)];
    env->observations[9] = (int16_t)env->score;
    env->observations[10] = (int16_t)env->length;
}

void snake_reset_fixed(SnakeEnv* env) {
    env->length = 3;
    env->head_r = 6;
    env->head_c = 5;
    env->body_r[0] = 6;
    env->body_c[0] = 5;
    env->body_r[1] = 6;
    env->body_c[1] = 4;
    env->body_r[2] = 6;
    env->body_c[2] = 3;
    env->food_r = 6;
    env->food_c = 8;
    env->dir = SNAKE_RIGHT;
    env->score = 0;
    env->tick = 0;
    env->terminal = 0;
    snake_rebuild_grid(env);
    snake_compute_observations(env);
}

void snake_set_food(SnakeEnv* env, int row, int col) {
    env->food_r = row;
    env->food_c = col;
    snake_rebuild_grid(env);
    snake_compute_observations(env);
}

static uint8_t snake_collides(const SnakeEnv* env, int r, int c) {
    uint8_t tile = env->grid[snake_index(env, r, c)];
    return tile == SNAKE_WALL || tile == SNAKE_BODY;
}

static void snake_spawn_next_food(SnakeEnv* env) {
    for (int r = 1; r < env->height - 1; r++) {
        for (int c = 1; c < env->width - 1; c++) {
            if (env->grid[snake_index(env, r, c)] == SNAKE_EMPTY) {
                env->food_r = r;
                env->food_c = c;
                return;
            }
        }
    }
}

int snake_step_env(SnakeEnv* env, uint8_t action, float* reward, uint8_t* terminal) {
    if (action >= SNAKE_ACTION_COUNT) {
        return TKM_ERR;
    }
    if (env->terminal) {
        *reward = 0.0f;
        *terminal = 1;
        return TKM_OK;
    }

    int nr = env->head_r + snake_dr(action);
    int nc = env->head_c + snake_dc(action);
    uint8_t grow = (nr == env->food_r && nc == env->food_c) ? 1u : 0u;
    uint8_t moves_into_old_tail = snake_action_moves_into_old_tail(env, action);

    *reward = 0.0f;
    *terminal = 0;

    if (snake_collides(env, nr, nc) && !moves_into_old_tail) {
        env->terminal = 1;
        *terminal = 1;
        *reward = -1.0f;
        snake_compute_observations(env);
        return TKM_OK;
    }

    int new_length = env->length + (grow ? 1 : 0);
    if (new_length > SNAKE_MAX_CELLS) {
        return TKM_ERR;
    }

    for (int i = new_length - 1; i > 0; i--) {
        env->body_r[i] = env->body_r[i - 1];
        env->body_c[i] = env->body_c[i - 1];
    }
    env->body_r[0] = (int16_t)nr;
    env->body_c[0] = (int16_t)nc;
    env->length = new_length;
    env->head_r = nr;
    env->head_c = nc;
    env->dir = action;
    env->tick++;

    if (grow) {
        env->score++;
        *reward = 1.0f;
    }

    snake_rebuild_grid(env);
    if (grow) {
        snake_spawn_next_food(env);
        snake_rebuild_grid(env);
    }
    snake_compute_observations(env);
    return TKM_OK;
}

int snake_step(SnakeEnv* env, uint8_t action) {
    float reward = 0.0f;
    uint8_t terminal = 0;
    return snake_step_env(env, action, &reward, &terminal);
}

int snake_write_features(const SnakeEnv* env, float* out_features, uint32_t max_features) {
    if (!env || !out_features || max_features < SNAKE_FEATURE_COUNT) {
        return TKM_ERR;
    }

    out_features[0] = (float)env->head_r / (float)(SNAKE_HEIGHT - 1);
    out_features[1] = (float)env->head_c / (float)(SNAKE_WIDTH - 1);
    out_features[2] = (float)env->food_r / (float)(SNAKE_HEIGHT - 1);
    out_features[3] = (float)env->food_c / (float)(SNAKE_WIDTH - 1);
    out_features[4] = (float)(env->food_r - env->head_r) / (float)(SNAKE_HEIGHT - 1);
    out_features[5] = (float)(env->food_c - env->head_c) / (float)(SNAKE_WIDTH - 1);
    out_features[6] = env->dir == SNAKE_UP ? 1.0f : 0.0f;
    out_features[7] = env->dir == SNAKE_DOWN ? 1.0f : 0.0f;
    out_features[8] = env->dir == SNAKE_LEFT ? 1.0f : 0.0f;
    out_features[9] = env->dir == SNAKE_RIGHT ? 1.0f : 0.0f;
    out_features[10] = (float)env->length / (float)SNAKE_MAX_CELLS;
    out_features[11] = env->terminal ? 1.0f : 0.0f;

    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            uint8_t tile = env->grid[snake_index(env, r, c)];
            float value = 0.0f;
            if (tile == SNAKE_BODY) {
                value = 1.0f;
            } else if (tile == SNAKE_FOOD) {
                value = -1.0f;
            }
            out_features[12 + snake_index(env, r, c)] = value;
        }
    }

    return TKM_OK;
}

int snake_write_action_features(const SnakeEnv* env, float* out_features, uint32_t max_features) {
    int current_distance;
    float distance_scale;

    if (!env || !out_features || max_features < SNAKE_ACTION_FEATURE_COUNT) {
        return TKM_ERR;
    }
    if (snake_write_features(env, out_features, max_features) != TKM_OK) {
        return TKM_ERR;
    }

    current_distance =
        snake_abs_int(env->food_r - env->head_r) +
        snake_abs_int(env->food_c - env->head_c);
    distance_scale = (float)(SNAKE_WIDTH + SNAKE_HEIGHT);

    for (uint32_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        uint8_t action_u8 = (uint8_t)action;
        int next_r = env->head_r + snake_dr(action_u8);
        int next_c = env->head_c + snake_dc(action_u8);
        int next_distance =
            snake_abs_int(env->food_r - next_r) +
            snake_abs_int(env->food_c - next_c);
        uint32_t base = SNAKE_FEATURE_COUNT + action * SNAKE_ACTION_FEATURES_PER_ACTION;

        out_features[base + 0u] = snake_action_terminals(env, action_u8) ? 0.0f : 1.0f;
        out_features[base + 1u] = (float)(next_distance - current_distance) / distance_scale;
        out_features[base + 2u] = (float)(env->food_r - next_r) / (float)(SNAKE_HEIGHT - 1);
        out_features[base + 3u] = (float)(env->food_c - next_c) / (float)(SNAKE_WIDTH - 1);
    }

    return TKM_OK;
}

int snake_write_action_space_features(const SnakeEnv* env, float* out_features, uint32_t max_features) {
    int current_distance;
    float distance_scale;

    if (!env || !out_features || max_features < SNAKE_ACTION_SPACE_FEATURE_COUNT) {
        return TKM_ERR;
    }
    if (snake_write_features(env, out_features, max_features) != TKM_OK) {
        return TKM_ERR;
    }

    current_distance =
        snake_abs_int(env->food_r - env->head_r) +
        snake_abs_int(env->food_c - env->head_c);
    distance_scale = (float)(SNAKE_WIDTH + SNAKE_HEIGHT);

    for (uint32_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        uint8_t action_u8 = (uint8_t)action;
        int next_r = env->head_r + snake_dr(action_u8);
        int next_c = env->head_c + snake_dc(action_u8);
        int next_distance =
            snake_abs_int(env->food_r - next_r) +
            snake_abs_int(env->food_c - next_c);
        uint8_t action_terminals = snake_action_terminals(env, action_u8);
        float reachable_fraction = 0.0f;
        uint32_t base = SNAKE_FEATURE_COUNT + action * SNAKE_ACTION_SPACE_FEATURES_PER_ACTION;

        if (!action_terminals) {
            SnakeEnv next = *env;
            float reward = 0.0f;
            uint8_t terminal = 0;
            if (snake_step_env(&next, action_u8, &reward, &terminal) == TKM_OK && !terminal) {
                reachable_fraction = (float)snake_reachable_count(&next, next.head_r, next.head_c) /
                    (float)SNAKE_MAX_CELLS;
            }
        }

        out_features[base + 0u] = action_terminals ? 0.0f : 1.0f;
        out_features[base + 1u] = (float)(next_distance - current_distance) / distance_scale;
        out_features[base + 2u] = (float)(env->food_r - next_r) / (float)(SNAKE_HEIGHT - 1);
        out_features[base + 3u] = (float)(env->food_c - next_c) / (float)(SNAKE_WIDTH - 1);
        out_features[base + 4u] = reachable_fraction;
    }

    return TKM_OK;
}

int snake_write_action_food_space_features(const SnakeEnv* env, float* out_features, uint32_t max_features) {
    int current_distance;
    float distance_scale;

    if (!env || !out_features || max_features < SNAKE_ACTION_FOOD_SPACE_FEATURE_COUNT) {
        return TKM_ERR;
    }
    if (snake_write_features(env, out_features, max_features) != TKM_OK) {
        return TKM_ERR;
    }

    current_distance =
        snake_abs_int(env->food_r - env->head_r) +
        snake_abs_int(env->food_c - env->head_c);
    distance_scale = (float)(SNAKE_WIDTH + SNAKE_HEIGHT);

    for (uint32_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        uint8_t action_u8 = (uint8_t)action;
        int next_r = env->head_r + snake_dr(action_u8);
        int next_c = env->head_c + snake_dc(action_u8);
        int next_distance =
            snake_abs_int(env->food_r - next_r) +
            snake_abs_int(env->food_c - next_c);
        uint8_t action_terminals = snake_action_terminals(env, action_u8);
        float reachable_fraction = 0.0f;
        float eats_food = (next_r == env->food_r && next_c == env->food_c) ? 1.0f : 0.0f;
        uint32_t base = SNAKE_FEATURE_COUNT + action * SNAKE_ACTION_FOOD_SPACE_FEATURES_PER_ACTION;

        if (!action_terminals) {
            SnakeEnv next = *env;
            float reward = 0.0f;
            uint8_t terminal = 0;
            if (snake_step_env(&next, action_u8, &reward, &terminal) == TKM_OK && !terminal) {
                reachable_fraction = (float)snake_reachable_count(&next, next.head_r, next.head_c) /
                    (float)SNAKE_MAX_CELLS;
            }
        }

        out_features[base + 0u] = action_terminals ? 0.0f : 1.0f;
        out_features[base + 1u] = (float)(next_distance - current_distance) / distance_scale;
        out_features[base + 2u] = (float)(env->food_r - next_r) / (float)(SNAKE_HEIGHT - 1);
        out_features[base + 3u] = (float)(env->food_c - next_c) / (float)(SNAKE_WIDTH - 1);
        out_features[base + 4u] = reachable_fraction;
        out_features[base + 5u] = eats_food;
    }

    return TKM_OK;
}

static uint8_t snake_action_moves_into_old_tail(const SnakeEnv* env, uint8_t action) {
    int nr;
    int nc;

    if (!env || action >= SNAKE_ACTION_COUNT || env->length <= 0) {
        return 0;
    }

    nr = env->head_r + snake_dr(action);
    nc = env->head_c + snake_dc(action);
    if (nr == env->food_r && nc == env->food_c) {
        return 0;
    }

    return (nr == env->body_r[env->length - 1] &&
        nc == env->body_c[env->length - 1]) ? 1u : 0u;
}

static uint8_t snake_action_terminals(const SnakeEnv* env, uint8_t action) {
    SnakeEnv next;
    float reward = 0.0f;
    uint8_t terminal = 0;

    if (!env || action >= SNAKE_ACTION_COUNT) {
        return 1;
    }

    next = *env;
    if (snake_step_env(&next, action, &reward, &terminal) != TKM_OK) {
        return 1;
    }

    return terminal ? 1u : 0u;
}

int snake_write_action_mask_immediate(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions) {
    if (!env || !out_mask || max_actions < SNAKE_ACTION_COUNT) {
        return TKM_ERR;
    }

    for (uint8_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        out_mask[action] = snake_action_terminals(env, action) ? 0u : 1u;
    }

    return TKM_OK;
}

int snake_write_action_mask_survival(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions) {
    uint8_t immediate_mask[SNAKE_ACTION_COUNT];
    uint8_t survival_mask[SNAKE_ACTION_COUNT];
    uint8_t has_survival_action = 0;

    if (!env || !out_mask || max_actions < SNAKE_ACTION_COUNT) {
        return TKM_ERR;
    }

    for (uint8_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        immediate_mask[action] = snake_action_terminals(env, action) ? 0u : 1u;
        survival_mask[action] = snake_action_keeps_space(env, action, 2u);
        if (survival_mask[action]) {
            has_survival_action = 1;
        }
    }

    for (uint8_t action = 0; action < SNAKE_ACTION_COUNT; action++) {
        out_mask[action] = has_survival_action ? survival_mask[action] : immediate_mask[action];
    }

    return TKM_OK;
}

int snake_write_action_mask(const SnakeEnv* env, uint8_t* out_mask, uint32_t max_actions) {
    return snake_write_action_mask_survival(env, out_mask, max_actions);
}

uint8_t snake_filter_unsafe_action(const SnakeEnv* env, uint8_t suggested_action, uint8_t fallback_action) {
    if (!env) {
        return SNAKE_RIGHT;
    }
    if (!snake_action_terminals(env, suggested_action)) {
        return suggested_action;
    }
    if (!snake_action_terminals(env, fallback_action)) {
        return fallback_action;
    }
    return snake_first_safe_action(env);
}

uint8_t snake_filter_planner_score_action(const SnakeEnv* env, uint8_t suggested_action) {
    uint8_t planner_action;

    if (!env) {
        return SNAKE_RIGHT;
    }
    if (suggested_action >= SNAKE_ACTION_COUNT) {
        return snake_planner_action(env);
    }

    planner_action = snake_planner_action(env);
    if (snake_score_action(env, planner_action) > snake_score_action(env, suggested_action)) {
        return planner_action;
    }

    return suggested_action;
}

uint8_t snake_filter_best_score_action(const SnakeEnv* env) {
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};
    int best_score = -1;
    uint8_t best_action;

    if (!env) {
        return SNAKE_RIGHT;
    }

    best_action = snake_first_safe_action(env);
    for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
        uint8_t action = actions[i];
        int score = snake_score_action(env, action);

        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

static int snake_rollout_score_action(
    const SnakeEnv* env,
    uint8_t action,
    uint32_t horizon,
    TkmDecoderRolloutScoreKind score_mode
) {
    SnakeEnv next;
    float reward = 0.0f;
    uint8_t terminal = 0;
    uint32_t steps = 0;
    uint32_t food = 0;

    if (!env || action >= SNAKE_ACTION_COUNT || horizon == 0u) {
        return -1000000;
    }

    next = *env;
    if (snake_step_env(&next, action, &reward, &terminal) != TKM_OK || terminal) {
        return -1000000;
    }

    steps++;
    if (reward > 0.0f) {
        food++;
    }

    while (steps < horizon && !next.terminal) {
        uint8_t planner_action = snake_planner_action(&next);
        reward = 0.0f;
        terminal = 0;
        if (snake_step_env(&next, planner_action, &reward, &terminal) != TKM_OK) {
            return -1000000;
        }
        steps++;
        if (reward > 0.0f) {
            food++;
        }
    }

    if (score_mode == TKM_DECODER_ROLLOUT_SCORE_SPACE_DISTANCE) {
        uint32_t reachable = next.terminal ? 0u : snake_reachable_count(&next, next.head_r, next.head_c);
        int distance = next.terminal ? (SNAKE_WIDTH + SNAKE_HEIGHT) : snake_food_distance_after(&next);
        return (int)(food * 10000u + (uint32_t)next.score * 100u + reachable * 20u + steps * 10u) -
            distance * 25;
    }

    return (int)(food * 10000u + steps * 10u + (uint32_t)next.score);
}

uint8_t snake_filter_rollout_score_mode_action(
    const SnakeEnv* env,
    uint8_t suggested_action,
    uint32_t horizon,
    TkmDecoderRolloutScoreKind score_mode
) {
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};
    int best_score;
    uint8_t best_action;

    if (!env) {
        return SNAKE_RIGHT;
    }
    if (horizon == 0u) {
        return snake_filter_planner_score_action(env, suggested_action);
    }

    if (suggested_action < SNAKE_ACTION_COUNT) {
        best_action = suggested_action;
    } else {
        best_action = snake_first_safe_action(env);
    }
    best_score = snake_rollout_score_action(env, best_action, horizon, score_mode);

    for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
        uint8_t action = actions[i];
        int score = snake_rollout_score_action(env, action, horizon, score_mode);

        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    if (best_score <= -1000000) {
        return snake_first_safe_action(env);
    }

    return best_action;
}

uint8_t snake_filter_rollout_score_action(const SnakeEnv* env, uint8_t suggested_action, uint32_t horizon) {
    return snake_filter_rollout_score_mode_action(
        env,
        suggested_action,
        horizon,
        TKM_DECODER_ROLLOUT_SCORE_FOOD_STEPS
    );
}

uint8_t snake_decode_action(const SnakeEnv* env, const TkmDecoderConfig* decoder, uint8_t suggested_action) {
    if (!env) {
        return SNAKE_RIGHT;
    }
    if (!decoder) {
        return suggested_action < SNAKE_ACTION_COUNT ? suggested_action : snake_first_safe_action(env);
    }

    if (decoder->fallback == TKM_DECODER_FALLBACK_PLANNER_SCORE) {
        return snake_filter_planner_score_action(env, suggested_action);
    }
    if (decoder->fallback == TKM_DECODER_FALLBACK_BEST_SCORE) {
        return snake_filter_best_score_action(env);
    }
    if (decoder->fallback == TKM_DECODER_FALLBACK_ROLLOUT_SCORE) {
        return snake_filter_rollout_score_mode_action(
            env,
            suggested_action,
            decoder->rollout_horizon,
            decoder->rollout_score_mode
        );
    }

    return suggested_action < SNAKE_ACTION_COUNT ? suggested_action : snake_first_safe_action(env);
}

static uint8_t snake_food_dir(const SnakeEnv* env) {
    int dr = env->food_r - env->head_r;
    int dc = env->food_c - env->head_c;
    int abs_dr = dr < 0 ? -dr : dr;
    int abs_dc = dc < 0 ? -dc : dc;

    if (abs_dc >= abs_dr) {
        return dc < 0 ? SNAKE_LEFT : SNAKE_RIGHT;
    }
    return dr < 0 ? SNAKE_UP : SNAKE_DOWN;
}

static uint8_t snake_danger_for_action(const SnakeEnv* env, uint8_t action) {
    int nr = env->head_r + snake_dr(action);
    int nc = env->head_c + snake_dc(action);
    return snake_collides(env, nr, nc);
}

uint16_t snake_tokenize_observation(const SnakeEnv* env) {
    uint8_t front = snake_danger_for_action(env, env->dir);
    uint8_t left = snake_danger_for_action(env, snake_left_of(env->dir));
    uint8_t right = snake_danger_for_action(env, snake_right_of(env->dir));
    uint8_t dangers = (uint8_t)(front | (left << 1) | (right << 2));
    return (uint16_t)(env->dir * 32u + snake_food_dir(env) * 8u + dangers);
}

uint16_t snake_tokenize_rich_observation(const SnakeEnv* env) {
    int head_r = env->head_r - 1;
    int head_c = env->head_c - 1;
    int food_r = env->food_r - 1;
    int food_c = env->food_c - 1;
    uint16_t head_index;
    uint16_t food_index;

    if (head_r < 0 || head_r >= 10 || head_c < 0 || head_c >= 10 ||
        food_r < 0 || food_r >= 10 || food_c < 0 || food_c >= 10) {
        return TKM_INVALID_TOKEN;
    }

    head_index = (uint16_t)(head_r * 10 + head_c);
    food_index = (uint16_t)(food_r * 10 + food_c);
    return (uint16_t)((uint16_t)env->dir * 10000u + head_index * 100u + food_index);
}

static uint32_t snake_hash_u32(uint32_t hash, uint32_t value) {
    for (uint32_t shift = 0; shift < 32u; shift += 8u) {
        hash ^= (value >> shift) & 0xffu;
        hash *= 16777619u;
    }
    return hash;
}

int32_t snake_tokenize_full_observation(const SnakeEnv* env) {
    uint32_t hash = 2166136261u;

    if (!env || env->length <= 0 || env->length > SNAKE_MAX_CELLS) {
        return -1;
    }

    hash = snake_hash_u32(hash, (uint32_t)env->head_r);
    hash = snake_hash_u32(hash, (uint32_t)env->head_c);
    hash = snake_hash_u32(hash, (uint32_t)env->food_r);
    hash = snake_hash_u32(hash, (uint32_t)env->food_c);
    hash = snake_hash_u32(hash, (uint32_t)env->dir);
    hash = snake_hash_u32(hash, (uint32_t)env->length);

    for (int i = 0; i < env->length; i++) {
        hash = snake_hash_u32(hash, (uint32_t)env->body_r[i]);
        hash = snake_hash_u32(hash, (uint32_t)env->body_c[i]);
    }

    return (int32_t)(hash & 0x7fffffffu);
}

int snake_policy_tokenizer_init(TkmIntBins* tokenizer) {
    return tkm_int_bins_init(tokenizer, 0, SNAKE_TOKEN_COUNT - 1);
}

int snake_rich_policy_tokenizer_init(TkmIntBins* tokenizer) {
    return tkm_int_bins_init(tokenizer, 0, (int32_t)(SNAKE_RICH_TOKEN_COUNT - 1u));
}

static uint8_t snake_first_safe_action(const SnakeEnv* env) {
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};
    for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
        uint8_t action = actions[i];
        int nr = env->head_r + snake_dr(action);
        int nc = env->head_c + snake_dc(action);
        if (!snake_collides(env, nr, nc)) {
            return action;
        }
    }
    return env->dir;
}

uint8_t snake_oracle_action(const SnakeEnv* env) {
    int visited[SNAKE_MAX_CELLS] = {0};
    uint8_t first_action[SNAKE_MAX_CELLS] = {0};
    SnakeCell queue[SNAKE_MAX_CELLS];
    int head = 0;
    int tail = 0;
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};

    queue[tail++] = (SnakeCell){env->head_r, env->head_c};
    visited[snake_index(env, env->head_r, env->head_c)] = 1;

    while (head < tail) {
        SnakeCell cell = queue[head++];
        for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
            uint8_t action = actions[i];
            int nr = cell.r + snake_dr(action);
            int nc = cell.c + snake_dc(action);
            int idx = snake_index(env, nr, nc);

            if (visited[idx] || snake_collides(env, nr, nc)) {
                continue;
            }

            visited[idx] = 1;
            first_action[idx] = (cell.r == env->head_r && cell.c == env->head_c) ?
                action : first_action[snake_index(env, cell.r, cell.c)];

            if (nr == env->food_r && nc == env->food_c) {
                return first_action[idx];
            }

            queue[tail++] = (SnakeCell){nr, nc};
        }
    }

    return snake_first_safe_action(env);
}

static uint32_t snake_reachable_count(const SnakeEnv* env, int start_r, int start_c) {
    int visited[SNAKE_MAX_CELLS] = {0};
    SnakeCell queue[SNAKE_MAX_CELLS];
    int head = 0;
    int tail = 0;
    uint32_t count = 0;
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};

    queue[tail++] = (SnakeCell){start_r, start_c};
    visited[snake_index(env, start_r, start_c)] = 1;

    while (head < tail) {
        SnakeCell cell = queue[head++];
        count++;

        for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
            int nr = cell.r + snake_dr(actions[i]);
            int nc = cell.c + snake_dc(actions[i]);
            int idx = snake_index(env, nr, nc);
            uint8_t tile = env->grid[idx];
            uint8_t is_start = (nr == start_r && nc == start_c) ? 1u : 0u;

            if (visited[idx]) {
                continue;
            }
            if (!is_start && (tile == SNAKE_WALL || tile == SNAKE_BODY)) {
                continue;
            }

            visited[idx] = 1;
            queue[tail++] = (SnakeCell){nr, nc};
        }
    }

    return count;
}

static int snake_food_distance_after(const SnakeEnv* env) {
    int dr = env->food_r - env->head_r;
    int dc = env->food_c - env->head_c;
    return (dr < 0 ? -dr : dr) + (dc < 0 ? -dc : dc);
}

static uint8_t snake_action_keeps_space(const SnakeEnv* env, uint8_t action, uint32_t extra_space) {
    SnakeEnv next;
    float reward = 0.0f;
    uint8_t terminal = 0;
    uint32_t reachable;
    uint32_t minimum_space;

    if (!env || action >= SNAKE_ACTION_COUNT) {
        return 0;
    }

    next = *env;
    if (snake_step_env(&next, action, &reward, &terminal) != TKM_OK || terminal) {
        return 0;
    }

    reachable = snake_reachable_count(&next, next.head_r, next.head_c);
    minimum_space = (uint32_t)next.length + extra_space;
    return reachable >= minimum_space ? 1u : 0u;
}

static int snake_score_action(const SnakeEnv* env, uint8_t action) {
    SnakeEnv next = *env;
    float reward = 0.0f;
    uint8_t terminal = 0;
    uint32_t reachable;
    uint32_t minimum_space;
    int distance;

    if (snake_step_env(&next, action, &reward, &terminal) != TKM_OK || terminal) {
        return -1;
    }

    reachable = snake_reachable_count(&next, next.head_r, next.head_c);
    minimum_space = (uint32_t)next.length + 2u;
    if (reachable < minimum_space) {
        return -1;
    }

    distance = snake_food_distance_after(&next);
    return (int)(reachable * 8u) + (reward > 0.0f ? 128 : 0) - distance;
}

uint8_t snake_planner_action(const SnakeEnv* env) {
    const uint8_t actions[SNAKE_ACTION_COUNT] = {SNAKE_UP, SNAKE_DOWN, SNAKE_LEFT, SNAKE_RIGHT};
    uint8_t oracle = snake_oracle_action(env);
    int oracle_score = snake_score_action(env, oracle);
    int best_score = -1;
    uint8_t best_action = env->dir;

    if (oracle_score >= 0) {
        return oracle;
    }

    for (uint8_t i = 0; i < SNAKE_ACTION_COUNT; i++) {
        int score = snake_score_action(env, actions[i]);
        if (score > best_score) {
            best_score = score;
            best_action = actions[i];
        }
    }

    if (best_score >= 0) {
        return best_action;
    }

    return snake_first_safe_action(env);
}

static int snake_collect_one(TkmTrajectory* trajectory, int food_r, int food_c) {
    SnakeEnv env;
    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < 32 && !env.terminal && env.score == 0; i++) {
        uint16_t token = snake_tokenize_observation(&env);
        uint8_t action = snake_oracle_action(&env);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_trajectory_append(trajectory, (int32_t)token, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }
    return TKM_OK;
}

int snake_collect_oracle_tokens(TkmTrajectory* trajectory) {
    const int foods[][2] = {
        {6, 8},
        {6, 2},
        {3, 5},
        {8, 5},
        {3, 8},
    };

    for (int repeat = 0; repeat < 3; repeat++) {
        for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
            if (snake_collect_one(trajectory, foods[i][0], foods[i][1]) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }
    return TKM_OK;
}

int snake_collect_planner_tokens(TkmTrajectory* trajectory, uint32_t max_steps) {
    return snake_collect_planner_tokens_for_food(trajectory, 6, 8, max_steps);
}

int snake_collect_planner_tokens_for_food(TkmTrajectory* trajectory, int food_r, int food_c, uint32_t max_steps) {
    SnakeEnv env;

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        int32_t token = snake_tokenize_full_observation(&env);
        uint8_t action = snake_planner_action(&env);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (token < 0) {
            return TKM_ERR;
        }
        if (snake_step_env(&env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_trajectory_append(trajectory, token, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int snake_run_token_policy(
    SnakeEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint16_t token = snake_tokenize_observation(env);
        uint8_t action = tkm_lookup_policy_predict(policy, token);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env->score;
        stats->terminal = terminal;

        if (stats->food_eaten > 0) {
            break;
        }
    }

    return TKM_OK;
}

int snake_run_rich_token_policy(
    SnakeEnv* env,
    const TkmLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint16_t token = snake_tokenize_rich_observation(env);
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (token == TKM_INVALID_TOKEN) {
            return TKM_ERR;
        }

        action = tkm_lookup_policy_predict(policy, token);
        if (snake_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env->score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

int snake_run_sparse_token_policy(
    SnakeEnv* env,
    const TkmSparseLookupPolicy* policy,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        int32_t token = snake_tokenize_full_observation(env);
        uint8_t action;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (token < 0) {
            return TKM_ERR;
        }

        action = tkm_sparse_lookup_policy_predict(policy, token);
        if (snake_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env->score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

int snake_run_decoder_config_policy(
    SnakeEnv* env,
    const TkmDecoderConfig* decoder,
    uint32_t max_steps,
    SnakeRolloutStats* stats
) {
    if (!env || !decoder || !stats) {
        return TKM_ERR;
    }

    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint8_t suggested_action = snake_planner_action(env);
        uint8_t action = snake_decode_action(env, decoder, suggested_action);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env->score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}

int snake_run_planner_policy(SnakeEnv* env, uint32_t max_steps, SnakeRolloutStats* stats) {
    memset(stats, 0, sizeof(*stats));

    for (uint32_t i = 0; i < max_steps && !env->terminal; i++) {
        uint8_t action = snake_planner_action(env);
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (snake_step_env(env, action, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }

        stats->steps++;
        if (reward > 0.0f) {
            stats->food_eaten++;
        }
        stats->score = env->score;
        stats->terminal = terminal;
    }

    return TKM_OK;
}
