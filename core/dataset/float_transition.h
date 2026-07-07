#ifndef TKM_FLOAT_TRANSITION_H
#define TKM_FLOAT_TRANSITION_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_PUFFERLIB_BREAKOUT_OBS_DIM 118u
#define TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT 3u

#define TKM_FLOAT_TRANSITION_MAX_ROWS 131072u
#define TKM_FLOAT_TRANSITION_MAX_OBS 128u
#define TKM_FLOAT_TRANSITION_MAX_EPISODES 1024u
#define TKM_FLOAT_TRANSITION_MAX_TEXT 64u

typedef struct {
    uint32_t version;
    char env[TKM_FLOAT_TRANSITION_MAX_TEXT];
    char source[TKM_FLOAT_TRANSITION_MAX_TEXT];
    int32_t env_index;
    int32_t episode;
    int32_t t;
    uint32_t obs_dim;
    float obs[TKM_FLOAT_TRANSITION_MAX_OBS];
    uint8_t action;
    float reward;
    uint8_t terminal;
} TkmFloatTransition;

typedef struct {
    uint32_t row_count;
    uint32_t obs_dim;
    uint32_t action_count;
    TkmFloatTransition rows[TKM_FLOAT_TRANSITION_MAX_ROWS];
} TkmFloatTransitionDataset;

typedef struct {
    char env[TKM_FLOAT_TRANSITION_MAX_TEXT];
    char source[TKM_FLOAT_TRANSITION_MAX_TEXT];
    char exporter[TKM_FLOAT_TRANSITION_MAX_TEXT];
    uint32_t obs_dim;
    uint32_t action_count;
    uint32_t row_count;
    uint32_t episode_count;
    float min_return;
    float max_return;
    float mean_return;
} TkmFloatTransitionManifest;

typedef struct {
    int32_t env_index;
    int32_t episode;
    uint32_t row_count;
    float episode_return;
} TkmFloatEpisodeSummary;

void tkm_float_transition_dataset_init(TkmFloatTransitionDataset* dataset);
int tkm_float_transition_manifest_load(TkmFloatTransitionManifest* manifest, const char* path);
int tkm_float_transition_dataset_load_jsonl(TkmFloatTransitionDataset* dataset, const char* path);
int tkm_float_transition_episode_returns(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatEpisodeSummary* episodes,
    uint32_t max_episodes,
    uint32_t* out_episode_count
);
int tkm_float_transition_filter_all(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out
);
int tkm_float_transition_filter_min_return(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    float min_return
);
int tkm_float_transition_filter_top_return(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    uint32_t max_episodes
);
int tkm_float_transition_filter_return_range(
    const TkmFloatTransitionDataset* dataset,
    TkmFloatTransitionDataset* out,
    float min_return,
    float max_return
);

#endif
