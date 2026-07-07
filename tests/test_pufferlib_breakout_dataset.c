#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/dataset/float_transition.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void check_close(float actual, float expected) {
    float diff = actual - expected;
    if (diff < 0.0f) {
        diff = -diff;
    }
    if (diff >= 0.0001f) {
        fprintf(stderr, "float mismatch: actual=%f expected=%f diff=%f\n", actual, expected, diff);
        CHECK(diff < 0.0001f);
    }
}

static void write_text_file(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    CHECK(file != 0);
    CHECK(fputs(text, file) >= 0);
    CHECK(fclose(file) == 0);
}

static void write_row(FILE* file, int env_index, int episode, int t, int action, float reward, int terminal) {
    CHECK(fprintf(file,
        "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"train_export\","
        "\"env_index\":%d,\"episode\":%d,\"t\":%d,\"obs_dim\":118,\"obs\":[",
        env_index,
        episode,
        t
    ) > 0);
    for (int i = 0; i < 118; i++) {
        CHECK(fprintf(file, "%s%.1f", i == 0 ? "" : ",", (float)(episode * 1000 + t * 10 + i)) > 0);
    }
    CHECK(fprintf(file,
        "],\"action\":%d,\"reward\":%.1f,\"terminal\":%d}\n",
        action,
        reward,
        terminal
    ) > 0);
}

static void write_synthetic_dataset(const char* path) {
    FILE* file = fopen(path, "wb");
    CHECK(file != 0);
    write_row(file, 0, 10, 0, 0, 1.0f, 0);
    write_row(file, 0, 10, 1, 1, 2.0f, 1);
    write_row(file, 0, 20, 0, 2, 5.0f, 0);
    write_row(file, 0, 20, 1, 1, 5.0f, 1);
    write_row(file, 1, 10, 0, 2, 4.0f, 1);
    CHECK(fclose(file) == 0);
}

static void test_manifest_loader_reads_required_metadata(void) {
    const char* path = "/tmp/tkm_pufferlib_breakout_manifest.json";
    TkmFloatTransitionManifest manifest;

    write_text_file(path,
        "{"
        "\"env\":\"pufferlib_breakout\","
        "\"source\":\"demo_export\","
        "\"obs_dim\":118,"
        "\"action_count\":3,"
        "\"row_count\":12,"
        "\"episode_count\":2,"
        "\"min_return\":1.0,"
        "\"max_return\":5.0,"
        "\"mean_return\":3.0,"
        "\"exporter\":\"tokamech_smoke\""
        "}\n"
    );

    CHECK(tkm_float_transition_manifest_load(&manifest, path) == TKM_OK);
    CHECK(strcmp(manifest.env, "pufferlib_breakout") == 0);
    CHECK(strcmp(manifest.source, "demo_export") == 0);
    CHECK(strcmp(manifest.exporter, "tokamech_smoke") == 0);
    CHECK(manifest.obs_dim == 118);
    CHECK(manifest.action_count == 3);
    CHECK(manifest.row_count == 12);
    CHECK(manifest.episode_count == 2);
    check_close(manifest.min_return, 1.0f);
    check_close(manifest.max_return, 5.0f);
    check_close(manifest.mean_return, 3.0f);
}

static void test_dataset_capacity_handles_large_exports(void) {
    CHECK(TKM_FLOAT_TRANSITION_MAX_ROWS >= 100000u);
}

static void test_manifest_loader_accepts_exporter_metadata_object(void) {
    const char* path = "/tmp/tkm_pufferlib_breakout_manifest_exporter_object.json";
    TkmFloatTransitionManifest manifest;

    write_text_file(path,
        "{"
        "\"env\":\"pufferlib_breakout\","
        "\"source\":\"train_export\","
        "\"version\":1,"
        "\"obs_dim\":118,"
        "\"action_count\":3,"
        "\"row_count\":512,"
        "\"episode_count\":64,"
        "\"min_return\":0,"
        "\"max_return\":0,"
        "\"mean_return\":0,"
        "\"exporter\":{"
        "\"jsonl_path\":\"/tmp/transitions-000000.jsonl\","
        "\"manifest_path\":\"/tmp/manifest.json\","
        "\"max_rows\":512"
        "}"
        "}\n"
    );

    CHECK(tkm_float_transition_manifest_load(&manifest, path) == TKM_OK);
    CHECK(strcmp(manifest.env, "pufferlib_breakout") == 0);
    CHECK(strcmp(manifest.source, "train_export") == 0);
    CHECK(strcmp(manifest.exporter, "metadata") == 0);
    CHECK(manifest.row_count == 512);
    CHECK(manifest.episode_count == 64);
}

static void test_jsonl_loader_validates_rows_and_preserves_fields(void) {
    const char* path = "/tmp/tkm_pufferlib_breakout_valid.jsonl";
    static TkmFloatTransitionDataset dataset;

    write_synthetic_dataset(path);

    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, path) == TKM_OK);
    CHECK(dataset.row_count == 5);
    CHECK(dataset.obs_dim == 118);
    CHECK(dataset.action_count == 3);
    CHECK(dataset.rows[0].env_index == 0);
    CHECK(dataset.rows[0].episode == 10);
    CHECK(dataset.rows[0].t == 0);
    CHECK(dataset.rows[0].action == 0);
    check_close(dataset.rows[0].reward, 1.0f);
    CHECK(dataset.rows[0].terminal == 0);
    CHECK(dataset.rows[1].terminal == 1);
    check_close(dataset.rows[2].obs[7], 20007.0f);
}

static void test_jsonl_loader_rejects_invalid_rows(void) {
    static TkmFloatTransitionDataset dataset;

    write_text_file("/tmp/tkm_pufferlib_breakout_bad_dim.jsonl",
        "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"train_export\","
        "\"env_index\":0,\"episode\":0,\"t\":0,\"obs_dim\":117,\"obs\":[0.0],"
        "\"action\":0,\"reward\":0.0,\"terminal\":0}\n"
    );
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, "/tmp/tkm_pufferlib_breakout_bad_dim.jsonl") == TKM_ERR);

    write_text_file("/tmp/tkm_pufferlib_breakout_bad_obs.jsonl",
        "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"train_export\","
        "\"env_index\":0,\"episode\":0,\"t\":0,\"obs_dim\":118,\"obs\":[0.0],"
        "\"action\":0,\"reward\":0.0,\"terminal\":0}\n"
    );
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, "/tmp/tkm_pufferlib_breakout_bad_obs.jsonl") == TKM_ERR);

    write_text_file("/tmp/tkm_pufferlib_breakout_bad_action.jsonl",
        "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"train_export\","
        "\"env_index\":0,\"episode\":0,\"t\":0,\"obs_dim\":118,\"obs\":["
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        "],\"action\":3,\"reward\":0.0,\"terminal\":0}\n"
    );
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, "/tmp/tkm_pufferlib_breakout_bad_action.jsonl") == TKM_ERR);

    write_text_file("/tmp/tkm_pufferlib_breakout_missing_reward.jsonl",
        "{\"version\":1,\"env\":\"pufferlib_breakout\",\"source\":\"train_export\","
        "\"env_index\":0,\"episode\":0,\"t\":0,\"obs_dim\":118,\"obs\":[0.0],"
        "\"action\":0,\"terminal\":0}\n"
    );
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, "/tmp/tkm_pufferlib_breakout_missing_reward.jsonl") == TKM_ERR);
}

static void test_episode_return_reconstruction_uses_env_index_and_episode(void) {
    const char* path = "/tmp/tkm_pufferlib_breakout_returns.jsonl";
    static TkmFloatTransitionDataset dataset;
    TkmFloatEpisodeSummary episodes[4];
    uint32_t episode_count = 0;

    write_synthetic_dataset(path);
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, path) == TKM_OK);
    CHECK(tkm_float_transition_episode_returns(&dataset, episodes, 4, &episode_count) == TKM_OK);
    CHECK(episode_count == 3);
    CHECK(episodes[0].env_index == 0);
    CHECK(episodes[0].episode == 10);
    check_close(episodes[0].episode_return, 3.0f);
    CHECK(episodes[1].env_index == 0);
    CHECK(episodes[1].episode == 20);
    check_close(episodes[1].episode_return, 10.0f);
    CHECK(episodes[2].env_index == 1);
    CHECK(episodes[2].episode == 10);
    check_close(episodes[2].episode_return, 4.0f);
}

static void test_filters_select_complete_episodes_by_return(void) {
    const char* path = "/tmp/tkm_pufferlib_breakout_filters.jsonl";
    static TkmFloatTransitionDataset dataset;
    static TkmFloatTransitionDataset filtered;

    write_synthetic_dataset(path);
    CHECK(tkm_float_transition_dataset_load_jsonl(&dataset, path) == TKM_OK);

    CHECK(tkm_float_transition_filter_min_return(&dataset, &filtered, 4.0f) == TKM_OK);
    CHECK(filtered.row_count == 3);
    CHECK(filtered.rows[0].episode == 20);
    CHECK(filtered.rows[1].episode == 20);
    CHECK(filtered.rows[2].env_index == 1);
    CHECK(filtered.rows[2].episode == 10);

    CHECK(tkm_float_transition_filter_top_return(&dataset, &filtered, 1) == TKM_OK);
    CHECK(filtered.row_count == 2);
    CHECK(filtered.rows[0].episode == 20);
    CHECK(filtered.rows[1].episode == 20);

    CHECK(tkm_float_transition_filter_return_range(&dataset, &filtered, 3.5f, 4.5f) == TKM_OK);
    CHECK(filtered.row_count == 1);
    CHECK(filtered.rows[0].env_index == 1);
    CHECK(filtered.rows[0].episode == 10);
}

int main(void) {
    test_manifest_loader_reads_required_metadata();
    test_dataset_capacity_handles_large_exports();
    test_manifest_loader_accepts_exporter_metadata_object();
    test_jsonl_loader_validates_rows_and_preserves_fields();
    test_jsonl_loader_rejects_invalid_rows();
    test_episode_return_reconstruction_uses_env_index_and_episode();
    test_filters_select_complete_episodes_by_return();
    puts("pufferlib breakout dataset tests passed");
    return 0;
}
