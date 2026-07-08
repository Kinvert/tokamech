#ifndef TKM_CENTERLINE_H
#define TKM_CENTERLINE_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/dataset/trajectory.h"

typedef enum {
    CENTERLINE_ACTION_LEFT = 0,
    CENTERLINE_ACTION_STAY = 1,
    CENTERLINE_ACTION_RIGHT = 2,
    CENTERLINE_ACTION_COUNT = 3,
} CenterlineAction;

typedef struct {
    int16_t start_offset;
    int16_t limit;
    uint32_t horizon;
} CenterlineConfig;

typedef struct {
    int16_t offset;
    int16_t limit;
    uint32_t horizon;
    uint32_t step;
    uint8_t terminal;
} CenterlineEnv;

void centerline_env_init(CenterlineEnv* env, int16_t start_offset, int16_t limit, uint32_t horizon);
int centerline_observe_i16(void* env, int16_t* obs);
uint8_t centerline_exploration_action(uint32_t step);
int centerline_action_to_command(uint8_t action, int8_t* command);
int centerline_step_command(void* env, int8_t command, float* reward, uint8_t* terminal);
int centerline_collect_exploration(const CenterlineConfig* cfg, TkmTrajectory* trajectory);

#endif
