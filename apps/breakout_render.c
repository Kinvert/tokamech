#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"

#include "core/dataset/trajectory.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/breakout/breakout.h"

static const Color BRICK_COLORS[BREAKOUT_BRICK_ROWS] = {
    {230, 41, 55, 255},
    {255, 161, 0, 255},
    {253, 249, 0, 255},
    {0, 228, 48, 255},
    {102, 191, 255, 255},
    {0, 121, 241, 255},
};

static void fail_if(int status, const char* label) {
    if (status != TKM_OK) {
        fprintf(stderr, "%s failed\n", label);
        exit(1);
    }
}

static const char* action_name(uint8_t action) {
    switch (action) {
        case BREAKOUT_LEFT:
            return "LEFT";
        case BREAKOUT_RIGHT:
            return "RIGHT";
        case BREAKOUT_NOOP:
            return "NOOP";
        default:
            return "UNKNOWN";
    }
}

static void draw_breakout(const BreakoutEnv* env, uint8_t action, uint8_t human_control) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(
        (int)env->paddle_x,
        (int)env->paddle_y,
        (int)env->paddle_width,
        (int)env->paddle_height,
        (Color){0, 255, 255, 255}
    );

    DrawRectangle(
        (int)env->ball_x,
        (int)env->ball_y,
        env->ball_width,
        env->ball_height,
        RAYWHITE
    );

    for (int row = 0; row < env->brick_rows; row++) {
        for (int col = 0; col < env->brick_cols; col++) {
            int idx = row * env->brick_cols + col;
            if (env->brick_states[idx] == 1.0f) {
                continue;
            }
            DrawRectangle(
                (int)env->brick_x[idx],
                (int)env->brick_y[idx],
                env->brick_width,
                env->brick_height,
                BRICK_COLORS[row]
            );
        }
    }

    DrawText(TextFormat("Score: %d", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %d", env->num_balls), env->width - 90, 10, 20, WHITE);
    DrawText(TextFormat("Action: %s", action_name(action)), 10, env->height - 28, 20, human_control ? ORANGE : SKYBLUE);
    DrawText("Hold SHIFT for human control. A/D or arrows move. R resets. ESC closes.", 10, env->height - 52, 14, GRAY);

    if (env->terminal) {
        DrawText("terminal - press R", env->width / 2 - 90, env->height / 2, 24, YELLOW);
    }

    EndDrawing();
}

int main(void) {
    TkmTrajectory dataset;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    BreakoutEnv env;
    uint8_t last_action = BREAKOUT_NOOP;
    int frame = 0;

    tkm_trajectory_init(&dataset);
    fail_if(breakout_collect_exploration_tokens(&dataset), "breakout_collect_exploration_tokens");
    fail_if(breakout_policy_tokenizer_init(&tokenizer), "breakout_policy_tokenizer_init");
    fail_if(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, BREAKOUT_ACTION_COUNT), "tkm_lookup_policy_train");

    breakout_env_init_default(&env);
    breakout_reset(&env);

    InitWindow(env.width, env.height, "Tokamech Breakout");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        uint8_t human_control = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

        if (IsKeyPressed(KEY_R)) {
            breakout_reset(&env);
            last_action = BREAKOUT_NOOP;
            frame = 0;
        }

        if (!env.terminal && frame % 4 == 0) {
            if (human_control) {
                last_action = BREAKOUT_NOOP;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    last_action = BREAKOUT_LEFT;
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    last_action = BREAKOUT_RIGHT;
                }
            } else {
                uint16_t token = breakout_tokenize_observation(env.observations);
                last_action = tkm_lookup_policy_predict(&policy, token);
            }
        }

        if (!env.terminal) {
            fail_if(breakout_step(&env, last_action), "breakout_step");
        }

        draw_breakout(&env, last_action, human_control);
        frame++;
    }

    CloseWindow();
    return 0;
}
