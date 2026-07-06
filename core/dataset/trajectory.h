#ifndef TKM_TRAJECTORY_H
#define TKM_TRAJECTORY_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_TRAJECTORY_MAX_STEPS 512u

typedef struct {
    int16_t obs_i16[TKM_TRAJECTORY_MAX_STEPS];
    uint8_t actions[TKM_TRAJECTORY_MAX_STEPS];
    float rewards[TKM_TRAJECTORY_MAX_STEPS];
    uint8_t terminals[TKM_TRAJECTORY_MAX_STEPS];
    uint32_t len;
} TkmTrajectory;

typedef struct {
    const TkmTrajectory* trajectory;
    uint32_t index;
} TkmTrajectoryCursor;

void tkm_trajectory_init(TkmTrajectory* trajectory);
int tkm_trajectory_append(TkmTrajectory* trajectory, int16_t obs, uint8_t action, float reward, uint8_t terminal);

void tkm_trajectory_cursor_init(TkmTrajectoryCursor* cursor, const TkmTrajectory* trajectory);
int tkm_trajectory_cursor_next(
    TkmTrajectoryCursor* cursor,
    int16_t* obs,
    uint8_t* action,
    float* reward,
    uint8_t* terminal
);

#endif
