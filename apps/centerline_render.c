#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"

#include "core/dataset/trajectory.h"
#include "core/decoder/discrete_action.h"
#include "core/runtime/runtime.h"
#include "core/tokenizer/int_bins.h"
#include "core/train/lookup_policy.h"
#include "projects/centerline/centerline.h"

#define SCREEN_W 960
#define SCREEN_H 540
#define CENTER_X (SCREEN_W / 2)
#define TRACK_Y 260
#define CELL_W 72
#define STEP_FRAMES 18

static void fail_if(int status, const char* label) {
    if (status != TKM_OK) {
        fprintf(stderr, "%s failed\n", label);
        exit(1);
    }
}

static void add_exploration_episode(TkmTrajectory* trajectory, int16_t start_offset) {
    CenterlineConfig cfg = {
        .start_offset = start_offset,
        .limit = 4,
        .horizon = 12,
    };

    fail_if(centerline_collect_exploration(&cfg, trajectory), "centerline_collect_exploration");
}

static const char* action_name(uint8_t action) {
    switch (action) {
        case CENTERLINE_ACTION_LEFT:
            return "LEFT";
        case CENTERLINE_ACTION_STAY:
            return "STAY";
        case CENTERLINE_ACTION_RIGHT:
            return "RIGHT";
        default:
            return "UNKNOWN";
    }
}

static void draw_centerline(const CenterlineEnv* env, uint8_t action, uint8_t human_control) {
    const Color bg = (Color){6, 24, 24, 255};
    const Color rail = (Color){55, 82, 82, 255};
    const Color center = (Color){0, 255, 255, 255};
    const Color bot = human_control ? ORANGE : RAYWHITE;
    int bot_x = CENTER_X + env->offset * CELL_W;

    BeginDrawing();
    ClearBackground(bg);

    DrawText("Tokamech centerline", 24, 24, 28, RAYWHITE);
    DrawText("Hold LEFT SHIFT for human control. Arrows/A/D move. R resets. ESC closes.", 24, 60, 18, GRAY);

    DrawLine(120, TRACK_Y, SCREEN_W - 120, TRACK_Y, rail);
    for (int i = -4; i <= 4; i++) {
        int x = CENTER_X + i * CELL_W;
        DrawLine(x, TRACK_Y - 26, x, TRACK_Y + 26, i == 0 ? center : rail);
        DrawText(TextFormat("%d", i), x - 8, TRACK_Y + 36, 18, i == 0 ? center : GRAY);
    }

    DrawCircle(bot_x, TRACK_Y, 28, bot);
    DrawCircleLines(bot_x, TRACK_Y, 32, center);
    DrawText(TextFormat("offset: %d", env->offset), 24, SCREEN_H - 112, 24, RAYWHITE);
    DrawText(TextFormat("step: %u / %u", env->step, env->horizon), 24, SCREEN_H - 80, 24, RAYWHITE);
    DrawText(TextFormat("action: %s", action_name(action)), 24, SCREEN_H - 48, 24, human_control ? ORANGE : center);

    if (env->terminal) {
        DrawText("terminal - press R to reset", CENTER_X - 150, 360, 24, YELLOW);
    }

    EndDrawing();
}

int main(void) {
    TkmTrajectory dataset;
    TkmTrajectory rollout;
    TkmIntBins tokenizer;
    TkmLookupPolicy policy;
    TkmDiscreteActionDecoder decoder;
    CenterlineEnv env;
    uint8_t last_action = CENTERLINE_ACTION_STAY;
    int frame = 0;

    tkm_trajectory_init(&dataset);
    tkm_trajectory_init(&rollout);
    fail_if(tkm_int_bins_init(&tokenizer, -4, 4), "tkm_int_bins_init");

    for (int16_t start = -4; start <= 4; start++) {
        add_exploration_episode(&dataset, start);
    }

    fail_if(tkm_lookup_policy_train(&policy, &tokenizer, &dataset, CENTERLINE_ACTION_COUNT), "tkm_lookup_policy_train");
    fail_if(tkm_discrete_action_decoder_init(&decoder), "tkm_discrete_action_decoder_init");
    fail_if(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_LEFT, -1), "decoder left");
    fail_if(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_STAY, 0), "decoder stay");
    fail_if(tkm_discrete_action_decoder_set(&decoder, CENTERLINE_ACTION_RIGHT, 1), "decoder right");

    centerline_env_init(&env, -4, 4, 240);

    InitWindow(SCREEN_W, SCREEN_H, "Tokamech Centerline");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        uint8_t human_control = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);

        if (IsKeyPressed(KEY_R)) {
            centerline_env_init(&env, -4, 4, 240);
            tkm_trajectory_init(&rollout);
            last_action = CENTERLINE_ACTION_STAY;
            frame = 0;
        }

        if (!env.terminal && frame % STEP_FRAMES == 0) {
            int16_t obs = 0;
            uint16_t token = 0;
            int8_t command = 0;
            float reward = 0.0f;
            uint8_t terminal = 0;

            fail_if(centerline_observe_i16(&env, &obs), "centerline_observe_i16");
            token = tkm_int_bins_encode(&tokenizer, obs);
            if (token == TKM_INVALID_TOKEN) {
                fprintf(stderr, "invalid token for obs %d\n", obs);
                break;
            }

            last_action = tkm_lookup_policy_predict(&policy, token);

            if (human_control) {
                last_action = CENTERLINE_ACTION_STAY;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                    last_action = CENTERLINE_ACTION_LEFT;
                }
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                    last_action = CENTERLINE_ACTION_RIGHT;
                }
            }

            fail_if(tkm_discrete_action_decode(&decoder, last_action, &command), "decode");
            fail_if(centerline_step_command(&env, command, &reward, &terminal), "centerline_step_command");
            fail_if(tkm_trajectory_append(&rollout, obs, last_action, reward, terminal), "trajectory append");
        }

        draw_centerline(&env, last_action, human_control);
        frame++;
    }

    CloseWindow();
    return 0;
}
