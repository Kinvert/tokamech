#include "projects/snake/snake.h"

#include <string.h>

typedef struct {
    int r;
    int c;
} SnakeCell;

static int snake_index(const SnakeEnv* env, int row, int col) {
    return row * env->width + col;
}

static int snake_in_bounds(const SnakeEnv* env, int row, int col) {
    return row >= 0 && row < env->height && col >= 0 && col < env->width;
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

void snake_env_init_default(SnakeEnv* env) {
    if (!env) {
        return;
    }
    memset(env, 0, sizeof(*env));
    env->width = SNAKE_WIDTH;
    env->height = SNAKE_HEIGHT;
}

void snake_compute_observations(SnakeEnv* env) {
    if (!env) {
        return;
    }
    env->observations[0] = (int16_t)env->head_r;
    env->observations[1] = (int16_t)env->head_c;
    env->observations[2] = (int16_t)env->food_r;
    env->observations[3] = (int16_t)env->food_c;
    env->observations[4] = (int16_t)env->dir;
    env->observations[5] = (int16_t)(env->food_r - env->head_r);
    env->observations[6] = (int16_t)(env->food_c - env->head_c);
    env->observations[7] = (int16_t)env->length;
    env->observations[8] = (int16_t)env->score;
    env->observations[9] = (int16_t)env->tick;
    env->observations[10] = (int16_t)env->terminal;
}

void snake_rebuild_grid(SnakeEnv* env) {
    if (!env) {
        return;
    }
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            uint8_t tile = (r == 0 || c == 0 || r == env->height - 1 || c == env->width - 1) ?
                SNAKE_WALL : SNAKE_EMPTY;
            env->grid[snake_index(env, r, c)] = tile;
        }
    }
    if (snake_in_bounds(env, env->food_r, env->food_c)) {
        env->grid[snake_index(env, env->food_r, env->food_c)] = SNAKE_FOOD;
    }
    for (int i = env->length - 1; i >= 0; i--) {
        if (snake_in_bounds(env, env->body_r[i], env->body_c[i])) {
            env->grid[snake_index(env, env->body_r[i], env->body_c[i])] = SNAKE_BODY;
        }
    }
    snake_compute_observations(env);
}

void snake_set_food(SnakeEnv* env, int row, int col) {
    if (!env) {
        return;
    }
    env->food_r = row;
    env->food_c = col;
    snake_rebuild_grid(env);
}

void snake_reset_fixed(SnakeEnv* env) {
    if (!env) {
        return;
    }
    env->width = SNAKE_WIDTH;
    env->height = SNAKE_HEIGHT;
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
}

static int snake_body_collision(const SnakeEnv* env, int row, int col, uint8_t growing) {
    int limit = env->length;
    if (!growing && limit > 0) {
        limit--;
    }
    for (int i = 0; i < limit; i++) {
        if (env->body_r[i] == row && env->body_c[i] == col) {
            return 1;
        }
    }
    return 0;
}

static void snake_place_next_food(SnakeEnv* env) {
    uint32_t seed = env->tick + (uint32_t)(env->score * 17 + env->head_r * 7 + env->head_c * 11);
    for (uint32_t offset = 0; offset < SNAKE_MAX_CELLS; offset++) {
        uint32_t pos = (seed + offset * 13u) % SNAKE_MAX_CELLS;
        int row = (int)(pos / (uint32_t)env->width);
        int col = (int)(pos % (uint32_t)env->width);
        if (row <= 0 || col <= 0 || row >= env->height - 1 || col >= env->width - 1) {
            continue;
        }
        if (!snake_body_collision(env, row, col, 1u)) {
            env->food_r = row;
            env->food_c = col;
            return;
        }
    }
}

int snake_step_env(SnakeEnv* env, uint8_t action, float* reward, uint8_t* terminal) {
    int next_r;
    int next_c;
    uint8_t growing;

    if (!env || !reward || !terminal || action >= SNAKE_ACTION_COUNT) {
        return TKM_ERR;
    }
    if (env->terminal) {
        *reward = 0.0f;
        *terminal = 1u;
        return TKM_OK;
    }

    next_r = env->head_r + snake_dr(action);
    next_c = env->head_c + snake_dc(action);
    growing = (next_r == env->food_r && next_c == env->food_c) ? 1u : 0u;
    *reward = 0.0f;
    *terminal = 0u;

    if (!snake_in_bounds(env, next_r, next_c) ||
        next_r == 0 || next_c == 0 || next_r == env->height - 1 || next_c == env->width - 1 ||
        snake_body_collision(env, next_r, next_c, growing)) {
        env->terminal = 1u;
        *terminal = 1u;
        *reward = -1.0f;
        snake_compute_observations(env);
        return TKM_OK;
    }

    if (growing && env->length < SNAKE_MAX_CELLS) {
        env->length++;
    }
    for (int i = env->length - 1; i > 0; i--) {
        env->body_r[i] = env->body_r[i - 1];
        env->body_c[i] = env->body_c[i - 1];
    }
    env->body_r[0] = (int16_t)next_r;
    env->body_c[0] = (int16_t)next_c;
    env->head_r = next_r;
    env->head_c = next_c;
    env->dir = action;
    env->tick++;

    if (growing) {
        env->score++;
        *reward = 1.0f;
        snake_place_next_food(env);
    }
    snake_rebuild_grid(env);
    return TKM_OK;
}

int snake_step(SnakeEnv* env, uint8_t action) {
    float reward = 0.0f;
    uint8_t terminal = 0u;
    return snake_step_env(env, action, &reward, &terminal);
}

uint16_t snake_tokenize_observation(const SnakeEnv* env) {
    uint32_t token;
    if (!env) {
        return 0u;
    }
    token = (uint32_t)(env->dir & 3u);
    token = token * 3u + (env->food_r < env->head_r ? 0u : (env->food_r > env->head_r ? 2u : 1u));
    token = token * 3u + (env->food_c < env->head_c ? 0u : (env->food_c > env->head_c ? 2u : 1u));
    token = token * 5u + (uint32_t)(env->length > 5 ? 4 : env->length - 1);
    return (uint16_t)(token % SNAKE_TOKEN_COUNT);
}

uint16_t snake_tokenize_rich_observation(const SnakeEnv* env) {
    uint32_t token;
    if (!env) {
        return TKM_INVALID_TOKEN;
    }
    token = (uint32_t)snake_tokenize_observation(env);
    token = token * (uint32_t)SNAKE_WIDTH + (uint32_t)env->head_c;
    token = token * (uint32_t)SNAKE_HEIGHT + (uint32_t)env->head_r;
    token = token * (uint32_t)SNAKE_WIDTH + (uint32_t)env->food_c;
    token = token * (uint32_t)SNAKE_HEIGHT + (uint32_t)env->food_r;
    return (uint16_t)(token % SNAKE_RICH_TOKEN_COUNT);
}

int32_t snake_tokenize_full_observation(const SnakeEnv* env) {
    uint32_t token;
    if (!env) {
        return -1;
    }
    token = (uint32_t)snake_tokenize_rich_observation(env);
    token = token * 31u + (uint32_t)(env->tick % 31u);
    token = token * 17u + (uint32_t)(env->score % 17u);
    return (int32_t)(token & 0x7fffffffu);
}

int snake_policy_tokenizer_init(TkmIntBins* tokenizer) {
    return tkm_int_bins_init(tokenizer, 0, (int32_t)(SNAKE_TOKEN_COUNT - 1u));
}

int snake_rich_policy_tokenizer_init(TkmIntBins* tokenizer) {
    return tkm_int_bins_init(tokenizer, 0, (int32_t)(SNAKE_RICH_TOKEN_COUNT - 1u));
}

int snake_write_features(const SnakeEnv* env, float* out_features, uint32_t max_features) {
    if (!env || !out_features || max_features < SNAKE_FEATURE_COUNT) {
        return TKM_ERR;
    }
    for (uint32_t i = 0; i < max_features; i++) {
        out_features[i] = 0.0f;
    }
    out_features[0] = (float)env->head_r / (float)(env->height - 1);
    out_features[1] = (float)env->head_c / (float)(env->width - 1);
    out_features[2] = (float)env->food_r / (float)(env->height - 1);
    out_features[3] = (float)env->food_c / (float)(env->width - 1);
    out_features[4] = env->food_r < env->head_r ? 1.0f : 0.0f;
    out_features[5] = env->food_c > env->head_c ? 1.0f : 0.0f;
    out_features[6] = env->food_r > env->head_r ? 1.0f : 0.0f;
    out_features[7] = env->food_c < env->head_c ? 1.0f : 0.0f;
    out_features[8] = env->terminal ? 1.0f : 0.0f;
    out_features[9] = 1.0f;
    out_features[10] = (float)env->length / (float)SNAKE_MAX_CELLS;
    out_features[11] = (float)env->score / 32.0f;
    for (uint32_t i = 0; i < SNAKE_MAX_CELLS; i++) {
        if (env->grid[i] == SNAKE_BODY) {
            out_features[12u + i] = 1.0f;
        } else if (env->grid[i] == SNAKE_FOOD) {
            out_features[12u + i] = -1.0f;
        }
    }
    return TKM_OK;
}


uint8_t snake_decode_action(const SnakeEnv* env, const TkmDecoderConfig* decoder, uint8_t suggested_action) {
    (void)env;
    (void)decoder;
    return suggested_action < SNAKE_ACTION_COUNT ? suggested_action : SNAKE_RIGHT;
}

uint8_t snake_exploration_action(uint32_t step) {
    const uint8_t pattern[] = {
        SNAKE_RIGHT, SNAKE_UP, SNAKE_RIGHT, SNAKE_DOWN,
        SNAKE_RIGHT, SNAKE_DOWN, SNAKE_LEFT, SNAKE_UP,
        SNAKE_RIGHT, SNAKE_RIGHT, SNAKE_UP, SNAKE_LEFT
    };
    return pattern[step % (sizeof(pattern) / sizeof(pattern[0]))];
}

int snake_collect_exploration_tokens_for_food(TkmTrajectory* trajectory, int food_r, int food_c, uint32_t max_steps) {
    SnakeEnv env;

    if (!trajectory || max_steps == 0u) {
        return TKM_ERR;
    }

    snake_env_init_default(&env);
    snake_reset_fixed(&env);
    snake_set_food(&env, food_r, food_c);

    for (uint32_t i = 0; i < max_steps && !env.terminal; i++) {
        int32_t token = (int32_t)snake_tokenize_observation(&env);
        uint8_t action = snake_exploration_action(i);
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

int snake_collect_exploration_tokens(TkmTrajectory* trajectory, uint32_t max_steps) {
    const int foods[][2] = {
        {6, 8},
        {6, 2},
        {3, 5},
        {8, 5},
        {3, 8},
    };

    if (!trajectory || max_steps == 0u) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < sizeof(foods) / sizeof(foods[0]); i++) {
        if (snake_collect_exploration_tokens_for_food(
                trajectory, foods[i][0], foods[i][1], max_steps) != TKM_OK) {
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
    if (!env || !policy || !stats) {
        return TKM_ERR;
    }
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
    if (!env || !policy || !stats) {
        return TKM_ERR;
    }
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
    if (!env || !policy || !stats) {
        return TKM_ERR;
    }
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
