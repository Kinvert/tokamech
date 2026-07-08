#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"

#include "core/config/run_config.h"
#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/snake/snake.h"

#define CELL_SIZE 42
#define HUD_H 56

static void fail_if(int status, const char* label) {
    if (status != TKM_OK) {
        fprintf(stderr, "%s failed\n", label);
        exit(1);
    }
}

static const char* action_name(uint8_t action) {
    switch (action) {
        case SNAKE_UP:
            return "UP";
        case SNAKE_DOWN:
            return "DOWN";
        case SNAKE_LEFT:
            return "LEFT";
        case SNAKE_RIGHT:
            return "RIGHT";
        default:
            return "UNKNOWN";
    }
}

static uint8_t human_action(uint8_t fallback) {
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        return SNAKE_UP;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        return SNAKE_DOWN;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        return SNAKE_LEFT;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        return SNAKE_RIGHT;
    }
    return fallback;
}

static const char* policy_name(uint8_t policy_mode) {
    if (policy_mode == 0u) {
        return "config";
    }
    if (policy_mode == 1u) {
        return "explore";
    }
    return "lookup";
}

static void draw_snake(const SnakeEnv* env, uint8_t action, uint8_t human_control, uint8_t policy_mode) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int x = c * CELL_SIZE;
            int y = HUD_H + r * CELL_SIZE;
            uint8_t tile = env->grid[r * env->width + c];
            Color color = (Color){18, 42, 42, 255};

            if (tile == SNAKE_WALL) {
                color = (Color){70, 90, 90, 255};
            } else if (tile == SNAKE_FOOD) {
                color = (Color){230, 41, 55, 255};
            } else if (tile == SNAKE_BODY) {
                color = (Color){0, 255, 255, 255};
            }

            DrawRectangle(x, y, CELL_SIZE - 1, CELL_SIZE - 1, color);
        }
    }

    DrawText("Tokamech Snake", 12, 10, 22, RAYWHITE);
    DrawText(TextFormat("Score: %d", env->score), 220, 14, 18, WHITE);
    DrawText(TextFormat("Action: %s", action_name(action)), 340, 14, 18, human_control ? ORANGE : SKYBLUE);
    DrawText(TextFormat("Policy: %s", policy_name(policy_mode)), 12, 34, 16, policy_mode == 0u ? GREEN : YELLOW);
    DrawText("SHIFT human. T cycles policy. R resets.", 190, 34, 16, GRAY);

    if (env->terminal) {
        DrawText("terminal - press R", 160, HUD_H + 220, 28, YELLOW);
    }

    EndDrawing();
}

int main(void) {
    TkmRunConfig run_config;
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    SnakeEnv env;
    uint8_t last_action = SNAKE_RIGHT;
    uint8_t policy_mode = 0;
    int frame = 0;

    fail_if(tkm_run_config_from_file("projects/snake/token_policy.ini", &run_config), "tkm_run_config_from_file");
    tkm_trajectory_init(&dataset);
    fail_if(snake_collect_exploration_tokens(&dataset, 64), "snake_collect_exploration_tokens");
    fail_if(snake_policy_tokenizer_init(&tokenizer), "snake_policy_tokenizer_init");
    fail_if(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, SNAKE_ACTION_COUNT), "tkm_lookup_policy_train");

    snake_env_init_default(&env);
    snake_reset_fixed(&env);

    InitWindow(SNAKE_WIDTH * CELL_SIZE, SNAKE_HEIGHT * CELL_SIZE + HUD_H, "Tokamech Snake");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        uint8_t human_control = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

        if (IsKeyPressed(KEY_R)) {
            snake_reset_fixed(&env);
            last_action = SNAKE_RIGHT;
            frame = 0;
        }
        if (IsKeyPressed(KEY_T)) {
            policy_mode = (uint8_t)((policy_mode + 1u) % 3u);
        }

        if (!env.terminal && frame % 10 == 0) {
            if (human_control) {
                last_action = human_action(last_action);
            } else if (policy_mode == 0u) {
                last_action = snake_decode_action(&env, &run_config.decoder, last_action);
            } else if (policy_mode == 1u) {
                last_action = snake_exploration_action((uint32_t)(frame / 10));
            } else {
                uint16_t token = snake_tokenize_observation(&env);
                last_action = tkm_lookup_policy_predict(&policy, token);
            }
            fail_if(snake_step(&env, last_action), "snake_step");
        }

        draw_snake(&env, last_action, human_control, policy_mode);
        frame++;
    }

    CloseWindow();
    return 0;
}
