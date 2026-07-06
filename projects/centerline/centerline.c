#include "projects/centerline/centerline.h"

static int16_t centerline_abs_i16(int16_t value) {
    return value < 0 ? (int16_t)-value : value;
}

void centerline_env_init(CenterlineEnv* env, int16_t start_offset, int16_t limit, uint32_t horizon) {
    env->offset = start_offset;
    env->limit = limit;
    env->horizon = horizon;
    env->step = 0;
    env->terminal = 0;
}

int centerline_observe_i16(void* env_ptr, int16_t* obs) {
    CenterlineEnv* env = (CenterlineEnv*)env_ptr;
    *obs = env->offset;
    return TKM_OK;
}

uint8_t centerline_oracle_action(int16_t offset) {
    if (offset < 0) {
        return CENTERLINE_ACTION_RIGHT;
    }
    if (offset > 0) {
        return CENTERLINE_ACTION_LEFT;
    }
    return CENTERLINE_ACTION_STAY;
}

int centerline_action_to_command(uint8_t action, int8_t* command) {
    switch (action) {
        case CENTERLINE_ACTION_LEFT:
            *command = -1;
            return TKM_OK;
        case CENTERLINE_ACTION_STAY:
            *command = 0;
            return TKM_OK;
        case CENTERLINE_ACTION_RIGHT:
            *command = 1;
            return TKM_OK;
        default:
            return TKM_ERR;
    }
}

int centerline_step_command(void* env_ptr, int8_t command, float* reward, uint8_t* terminal) {
    CenterlineEnv* env = (CenterlineEnv*)env_ptr;
    if (env->terminal) {
        *reward = 0.0f;
        *terminal = 1;
        return TKM_OK;
    }

    if (command < -1 || command > 1) {
        return TKM_ERR;
    }

    env->offset = (int16_t)(env->offset + command);
    env->step++;

    if (centerline_abs_i16(env->offset) > env->limit || env->step >= env->horizon) {
        env->terminal = 1;
    }

    *reward = env->offset == 0 ? 1.0f : 0.0f;
    *terminal = env->terminal;
    return TKM_OK;
}

int centerline_collect_oracle(const CenterlineConfig* cfg, TkmTrajectory* trajectory) {
    CenterlineEnv env;
    centerline_env_init(&env, cfg->start_offset, cfg->limit, cfg->horizon);

    for (uint32_t i = 0; i < cfg->horizon && !env.terminal; i++) {
        int16_t obs = 0;
        uint8_t action = 0;
        int8_t command = 0;
        float reward = 0.0f;
        uint8_t terminal = 0;

        if (centerline_observe_i16(&env, &obs) != TKM_OK) {
            return TKM_ERR;
        }

        action = centerline_oracle_action(obs);
        if (centerline_action_to_command(action, &command) != TKM_OK) {
            return TKM_ERR;
        }
        if (centerline_step_command(&env, command, &reward, &terminal) != TKM_OK) {
            return TKM_ERR;
        }
        if (tkm_trajectory_append(trajectory, obs, action, reward, terminal) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}
