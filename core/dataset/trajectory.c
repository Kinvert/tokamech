#include "core/dataset/trajectory.h"

void tkm_trajectory_init(TkmTrajectory* trajectory) {
    trajectory->len = 0;
}

int tkm_trajectory_append(TkmTrajectory* trajectory, int16_t obs, uint8_t action, float reward, uint8_t terminal) {
    if (trajectory->len >= TKM_TRAJECTORY_MAX_STEPS) {
        return TKM_ERR;
    }

    uint32_t i = trajectory->len;
    trajectory->obs_i16[i] = obs;
    trajectory->actions[i] = action;
    trajectory->rewards[i] = reward;
    trajectory->terminals[i] = terminal ? 1u : 0u;
    trajectory->len++;
    return TKM_OK;
}

void tkm_trajectory_cursor_init(TkmTrajectoryCursor* cursor, const TkmTrajectory* trajectory) {
    cursor->trajectory = trajectory;
    cursor->index = 0;
}

int tkm_trajectory_cursor_next(
    TkmTrajectoryCursor* cursor,
    int16_t* obs,
    uint8_t* action,
    float* reward,
    uint8_t* terminal
) {
    if (cursor->index >= cursor->trajectory->len) {
        return TKM_ERR;
    }

    uint32_t i = cursor->index;
    *obs = cursor->trajectory->obs_i16[i];
    *action = cursor->trajectory->actions[i];
    *reward = cursor->trajectory->rewards[i];
    *terminal = cursor->trajectory->terminals[i];
    cursor->index++;
    return TKM_OK;
}
