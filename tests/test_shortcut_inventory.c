#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void read_file(const char* path, char* buffer, size_t capacity) {
    FILE* file;
    size_t n;

    CHECK(path != 0);
    CHECK(buffer != 0);
    CHECK(capacity > 1u);

    file = fopen(path, "rb");
    CHECK(file != 0);
    n = fread(buffer, 1u, capacity - 1u, file);
    CHECK(ferror(file) == 0);
    CHECK(fclose(file) == 0);
    buffer[n] = '\0';
}

static void require_file_absent(const char* path) {
    FILE* file;

    CHECK(path != 0);
    file = fopen(path, "rb");
    if (file != 0) {
        (void)fclose(file);
        CHECK(file == 0);
    }
}

static void require_absent(const char* text, const char* needle) {
    CHECK(text != 0);
    CHECK(needle != 0);
    if (strstr(text, needle) != 0) {
        fprintf(stderr, "forbidden shortcut string still present: %s\n", needle);
        CHECK(strstr(text, needle) == 0);
    }
}

static void test_active_benchmark_source_has_no_snake_shortcut_symbols(void) {
    static char source[256000];

    read_file("apps/benchmark.c", source, sizeof(source));

    require_absent(source, "snake_mlp_ce_masked_holdout");
    require_absent(source, "snake_mlp_masked_ce_holdout");
    require_absent(source, "snake_mlp_dagger_masked_holdout");
    require_absent(source, "snake_mlp_dagger64_masked_holdout");
    require_absent(source, "snake_mlp_action_features_holdout");
    require_absent(source, "snake_action_scorer_holdout");
    require_absent(source, "snake_action_scorer_ce_holdout");
    require_absent(source, "snake_action_scorer_config_holdout");
    require_absent(source, "snake_action_scorer_config_file_holdout");
    require_absent(source, "snake_action_scorer_config_file_long_holdout");
}

static void test_default_benchmark_source_has_no_hand_coded_action_scorer_surfaces(void) {
    static char source[256000];

    read_file("apps/benchmark.c", source, sizeof(source));

    require_absent(source, "centerline_action_scorer");
    require_absent(source, "breakout_action_scorer");
}

static void test_default_benchmark_source_has_no_untrained_hard_coded_policy_entries(void) {
    static char benchmark_header[256000];
    static char benchmark_source[256000];
    static char benchmark_test[256000];

    read_file("apps/benchmark.h", benchmark_header, sizeof(benchmark_header));
    read_file("apps/benchmark.c", benchmark_source, sizeof(benchmark_source));
    read_file("tests/test_benchmark.c", benchmark_test, sizeof(benchmark_test));

    require_absent(benchmark_header, "tkm_bench_centerline_mlp(void)");
    require_absent(benchmark_source, "tkm_bench_centerline_mlp(void)");
    require_absent(benchmark_source, ".name = \"centerline_mlp\"");
    require_absent(benchmark_source, "tkm_print_bench_result(tkm_bench_centerline_mlp())");
    require_absent(benchmark_test, "tkm_bench_centerline_mlp()");
}

static void test_default_benchmark_source_has_no_direct_feature_to_action_baselines(void) {
    static const char* banned_entries[] = {
        "centerline_mlp_trained",
        "centerline_mlp_ce_trained",
        "centerline_nearest",
        "centerline_linear_policy",
        "breakout_nearest",
        "breakout_linear_policy",
        "breakout_mlp_ce_trained",
        "snake_nearest_multistart",
        "snake_mlp_ce_holdout",
        "snake_mlp_dagger_holdout",
    };
    static char benchmark_header[256000];
    static char benchmark_source[512000];
    static char benchmark_test[256000];

    read_file("apps/benchmark.h", benchmark_header, sizeof(benchmark_header));
    read_file("apps/benchmark.c", benchmark_source, sizeof(benchmark_source));
    read_file("tests/test_benchmark.c", benchmark_test, sizeof(benchmark_test));

    require_absent(benchmark_source, "core/model/linear_policy.h");
    require_absent(benchmark_source, "core/model/nearest_policy.h");
    require_absent(benchmark_source, "tkm_bench_one_hot");
    require_absent(benchmark_source, "tkm_bench_snake_mlp");
    for (size_t i = 0u; i < sizeof(banned_entries) / sizeof(banned_entries[0]); i++) {
        require_absent(benchmark_header, banned_entries[i]);
        require_absent(benchmark_source, banned_entries[i]);
        require_absent(benchmark_test, banned_entries[i]);
    }
}

static void test_generic_direct_action_policy_libraries_are_removed(void) {
    static char makefile[256000];

    read_file("Makefile", makefile, sizeof(makefile));

    require_absent(makefile, "NEAREST_POLICY_TEST_BIN");
    require_absent(makefile, "LINEAR_POLICY_TEST_BIN");
    require_absent(makefile, "core/model/nearest_policy.c");
    require_absent(makefile, "core/model/linear_policy.c");
    require_absent(makefile, "tests/test_nearest_policy.c");
    require_absent(makefile, "tests/test_linear_policy.c");

    require_file_absent("core/model/nearest_policy.h");
    require_file_absent("core/model/nearest_policy.c");
    require_file_absent("core/model/linear_policy.h");
    require_file_absent("core/model/linear_policy.c");
    require_file_absent("tests/test_nearest_policy.c");
    require_file_absent("tests/test_linear_policy.c");
}

static void test_pufferlib_benchmark_source_only_exposes_token_policy_methods(void) {
    static char source[256000];

    read_file("apps/pufferlib_breakout_benchmark.c", source, sizeof(source));

    require_absent(source, "BENCH_METHOD_MLP");
    require_absent(source, "BENCH_METHOD_NEAREST_WINDOW");
    require_absent(source, "BENCH_METHOD_SEQUENCE_CURSOR");
    require_absent(source, "BENCH_METHOD_INTERCEPT");
    require_absent(source, "nearest_window_");
    require_absent(source, "sequence_cursor_");
    require_absent(source, "intercept_");
    require_absent(source, "\"mlp_all\"");
    require_absent(source, "pufferlib_token_mlp.ini");
    require_absent(source, "token_linear_policy");
    require_absent(source, "token_action_mlp_context");
    require_absent(source, "token_mlp_delta_action");
}

static void test_pufferlib_render_source_only_exposes_token_policy_methods(void) {
    static char source[256000];

    read_file("apps/pufferlib_breakout_policy_render.c", source, sizeof(source));

    require_absent(source, "breakout_pufferlib_policy_load");
    require_absent(source, "breakout_pufferlib_policy_predict");
    require_absent(source, "breakout_pufferlib_nearest_predict");
    require_absent(source, "breakout_pufferlib_sequence_cursor");
    require_absent(source, "breakout_pufferlib_intercept_policy");
    require_absent(source, "TKM_PUFFERLIB_POLICY_CONFIG");
    require_absent(source, "TKM_PUFFERLIB_POLICY_PARAMS");
    require_absent(source, "nearest_window");
    require_absent(source, "sequence_cursor");
    require_absent(source, "intercept_rule");
    require_absent(source, "intercept_trained");
    require_absent(source, "pufferlib_token_mlp.ini");
    require_absent(source, "token_linear_policy");
    require_absent(source, "token_action_mlp_context");
    require_absent(source, "action_mlp_from_token_context");
}

static void test_pufferlib_smoke_source_evaluates_token_policy_not_direct_action_policy(void) {
    static char source[256000];

    read_file("apps/pufferlib_breakout_smoke.c", source, sizeof(source));

    require_absent(source, "BreakoutPufferlibPolicy");
    require_absent(source, "BreakoutPufferlibBcReport");
    require_absent(source, "breakout_pufferlib_policy_train_bc");
    require_absent(source, "breakout_pufferlib_policy_save");
    require_absent(source, "breakout_pufferlib_policy_load");
    require_absent(source, "breakout_pufferlib_policy_predict");
    require_absent(source, "pufferlib_token_mlp.ini");
}

static void test_pufferlib_data_paths_do_not_use_puffernet_teacher_models(void) {
    static char smoke_source[256000];
    static char makefile[256000];

    read_file("apps/pufferlib_breakout_smoke.c", smoke_source, sizeof(smoke_source));
    read_file("Makefile", makefile, sizeof(makefile));

    require_absent(smoke_source, "puffernet.h");
    require_absent(smoke_source, "PufferNet");
    require_absent(smoke_source, "load_weights");
    require_absent(smoke_source, "make_puffernet");
    require_absent(smoke_source, "forward_puffernet");
    require_absent(smoke_source, "free_puffernet");
    require_absent(smoke_source, "breakout_weights.bin");

    require_absent(makefile, "puffernet.h");
    require_absent(makefile, "breakout_weights.bin");
    require_absent(makefile, "PUFFERLIB_BREAKOUT_CHECKPOINT_EXPORT");
    require_absent(makefile, "pufferlib-breakout-checkpoint-export");
    require_absent(makefile, "pufferlib_breakout_checkpoint_export");
    require_absent(makefile, "apps/pufferlib_breakout_checkpoint_export.c");
}

static void test_pufferlib_policy_library_exposes_only_token_policy_path(void) {
    static char header[256000];
    static char source[256000];
    static char makefile[256000];

    read_file("projects/breakout/pufferlib_policy.h", header, sizeof(header));
    read_file("projects/breakout/pufferlib_policy.c", source, sizeof(source));
    read_file("Makefile", makefile, sizeof(makefile));

    require_absent(header, "BreakoutPufferlibPolicy");
    require_absent(header, "BreakoutPufferlibBcConfig");
    require_absent(header, "BreakoutPufferlibBcReport");
    require_absent(header, "BreakoutPufferlibSequenceCursor");
    require_absent(header, "BreakoutPufferlibIntercept");
    require_absent(header, "breakout_pufferlib_policy_");
    require_absent(header, "breakout_pufferlib_nearest_predict");
    require_absent(header, "breakout_pufferlib_sequence_cursor");
    require_absent(header, "breakout_pufferlib_intercept_policy");
    require_absent(header, "BREAKOUT_PUFFERLIB_TOKEN_MODEL_LINEAR_POLICY");
    require_absent(header, "BREAKOUT_PUFFERLIB_TOKEN_MODEL_ACTION_MLP_FROM_TOKEN_CONTEXT");
    require_absent(header, "BREAKOUT_PUFFERLIB_TOKEN_INPUT_DELTA_ACTION_HISTORY");

    require_absent(source, "BreakoutPufferlibPolicy");
    require_absent(source, "BreakoutPufferlibBcConfig");
    require_absent(source, "BreakoutPufferlibBcReport");
    require_absent(source, "BreakoutPufferlibSequenceCursor");
    require_absent(source, "BreakoutPufferlibIntercept");
    require_absent(source, "breakout_pufferlib_policy_");
    require_absent(source, "breakout_pufferlib_nearest_predict");
    require_absent(source, "breakout_pufferlib_sequence_cursor");
    require_absent(source, "breakout_pufferlib_intercept_policy");
    require_absent(source, "linear_policy");
    require_absent(source, "action_mlp_from_token_context");
    require_absent(source, "delta_action_history");
    require_absent(source, "breakout_token_action_mlp_pass");
    require_absent(source, "breakout_token_context_mlp_predict_action");

    require_absent(makefile, "pufferlib_token_mlp.ini");
    require_absent(makefile, "pufferlib_token_mlp_first_dims_h4.ini");
    require_absent(makefile, "pufferlib_token_mlp_first_dims24.ini");
    require_absent(makefile, "pufferlib_token_mlp_first_dims24_h4.ini");
    require_absent(makefile, "pufferlib_token_mlp_delta_action.ini");
    require_absent(makefile, "pufferlib_token_linear.ini");
    require_absent(makefile, "pufferlib_token_action_mlp_context.ini");
    require_absent(makefile, "pufferlib_token_action_mlp_first_dims.ini");
    require_absent(makefile, "pufferlib_token_action_mlp_manual.ini");

    require_file_absent("projects/breakout/pufferlib_token_mlp.ini");
    require_file_absent("projects/breakout/pufferlib_token_mlp_first_dims_h4.ini");
    require_file_absent("projects/breakout/pufferlib_token_mlp_first_dims24.ini");
    require_file_absent("projects/breakout/pufferlib_token_mlp_first_dims24_h4.ini");
    require_file_absent("projects/breakout/pufferlib_token_mlp_delta_action.ini");
    require_file_absent("projects/breakout/pufferlib_token_linear.ini");
    require_file_absent("projects/breakout/pufferlib_token_action_mlp_context.ini");
    require_file_absent("projects/breakout/pufferlib_token_action_mlp_first_dims.ini");
    require_file_absent("projects/breakout/pufferlib_token_action_mlp_manual.ini");
}

static void test_snake_sources_do_not_expose_oracle_or_planner_surfaces(void) {
    static char snake_header[256000];
    static char snake_source[256000];
    static char benchmark_header[256000];
    static char benchmark_source[256000];
    static char render_source[256000];
    static char snake_test[256000];
    static char benchmark_test[256000];

    read_file("projects/snake/snake.h", snake_header, sizeof(snake_header));
    read_file("projects/snake/snake.c", snake_source, sizeof(snake_source));
    read_file("apps/benchmark.h", benchmark_header, sizeof(benchmark_header));
    read_file("apps/benchmark.c", benchmark_source, sizeof(benchmark_source));
    read_file("apps/snake_render.c", render_source, sizeof(render_source));
    read_file("tests/test_snake_phase2.c", snake_test, sizeof(snake_test));
    read_file("tests/test_benchmark.c", benchmark_test, sizeof(benchmark_test));

    require_absent(snake_header, "snake_oracle_action");
    require_absent(snake_header, "snake_planner_action");
    require_absent(snake_header, "snake_collect_oracle_tokens");
    require_absent(snake_header, "snake_collect_planner_tokens");
    require_absent(snake_header, "snake_run_planner_policy");
    require_absent(snake_header, "snake_run_decoder_config_policy");
    require_absent(snake_header, "snake_write_action_mask");
    require_absent(snake_header, "SNAKE_ACTION_FEATURE");
    require_absent(snake_header, "snake_write_action_features");
    require_absent(snake_header, "snake_write_action_space_features");
    require_absent(snake_header, "snake_write_action_food_space_features");

    require_absent(snake_source, "snake_oracle_action");
    require_absent(snake_source, "snake_planner_action");
    require_absent(snake_source, "snake_collect_oracle_tokens");
    require_absent(snake_source, "snake_collect_planner_tokens");
    require_absent(snake_source, "snake_run_planner_policy");
    require_absent(snake_source, "snake_run_decoder_config_policy");
    require_absent(snake_source, "snake_write_action_mask");
    require_absent(snake_source, "snake_score_action");
    require_absent(snake_source, "snake_first_safe_action");
    require_absent(snake_source, "snake_write_action_features");
    require_absent(snake_source, "snake_write_action_space_features");
    require_absent(snake_source, "snake_write_action_food_space_features");

    require_absent(benchmark_header, "snake_planner");
    require_absent(benchmark_source, "snake_planner");
    require_absent(benchmark_source, "snake_collect_oracle_tokens");
    require_absent(benchmark_source, "snake_collect_planner_tokens");
    require_absent(benchmark_source, "snake_planner_action");
    require_absent(render_source, "snake_collect_oracle_tokens");
    require_absent(render_source, "snake_planner_action");
    require_absent(snake_test, "snake_oracle_action");
    require_absent(snake_test, "snake_collect_oracle_tokens");
    require_absent(snake_test, "snake_collect_planner_tokens");
    require_absent(snake_test, "snake_run_planner_policy");
    require_absent(snake_test, "snake_run_decoder_config_policy");
    require_absent(snake_test, "snake_write_action_mask");
    require_absent(benchmark_test, "snake_planner");
}

static void test_centerline_breakout_and_config_do_not_expose_oracle_surfaces(void) {
    static char centerline_header[256000];
    static char centerline_source[256000];
    static char breakout_header[256000];
    static char breakout_source[256000];
    static char benchmark_source[256000];
    static char centerline_render[256000];
    static char breakout_render[256000];
    static char centerline_test[256000];
    static char breakout_test[256000];
    static char layer_header[256000];
    static char layer_source[256000];
    static char factory_header[256000];
    static char factory_source[256000];
    static char layer_config_test[256000];
    static char layer_factory_test[256000];

    read_file("projects/centerline/centerline.h", centerline_header, sizeof(centerline_header));
    read_file("projects/centerline/centerline.c", centerline_source, sizeof(centerline_source));
    read_file("projects/breakout/breakout.h", breakout_header, sizeof(breakout_header));
    read_file("projects/breakout/breakout.c", breakout_source, sizeof(breakout_source));
    read_file("apps/benchmark.c", benchmark_source, sizeof(benchmark_source));
    read_file("apps/centerline_render.c", centerline_render, sizeof(centerline_render));
    read_file("apps/breakout_render.c", breakout_render, sizeof(breakout_render));
    read_file("tests/test_centerline_phase0.c", centerline_test, sizeof(centerline_test));
    read_file("tests/test_breakout_phase1.c", breakout_test, sizeof(breakout_test));
    read_file("core/config/layer_config.h", layer_header, sizeof(layer_header));
    read_file("core/config/layer_config.c", layer_source, sizeof(layer_source));
    read_file("core/config/layer_factory.h", factory_header, sizeof(factory_header));
    read_file("core/config/layer_factory.c", factory_source, sizeof(factory_source));
    read_file("tests/test_layer_config.c", layer_config_test, sizeof(layer_config_test));
    read_file("tests/test_layer_factory.c", layer_factory_test, sizeof(layer_factory_test));

    require_absent(centerline_header, "oracle");
    require_absent(centerline_source, "oracle");
    require_absent(breakout_header, "oracle");
    require_absent(breakout_source, "oracle");
    require_absent(benchmark_source, "centerline_collect_oracle");
    require_absent(benchmark_source, "breakout_collect_oracle");
    require_absent(centerline_render, "oracle");
    require_absent(breakout_render, "oracle");
    require_absent(centerline_test, "oracle");
    require_absent(breakout_test, "oracle");
    require_absent(layer_header, "PLANNER_ORACLE");
    require_absent(layer_source, "planner_oracle");
    require_absent(layer_header, "TKM_MODEL_MLP_WINDOW");
    require_absent(layer_header, "TKM_MODEL_NEAREST_POLICY");
    require_absent(layer_header, "TKM_MODEL_LINEAR_POLICY");
    require_absent(layer_source, "mlp_window");
    require_absent(layer_source, "nearest_policy");
    require_absent(layer_source, "linear_policy");
    require_absent(factory_header, "core/model/mlp_window.h");
    require_absent(factory_header, "tkm_layer_factory_init_mlp_window");
    require_absent(factory_source, "TKM_MODEL_MLP_WINDOW");
    require_absent(factory_source, "tkm_layer_factory_init_mlp_window");
    require_absent(layer_config_test, "mlp_window");
    require_absent(layer_config_test, "nearest_policy");
    require_absent(layer_config_test, "linear_policy");
    require_absent(layer_factory_test, "kind = mlp_window");
    require_absent(layer_factory_test, "tkm_layer_factory_init_mlp_window");
}

static void test_decoder_and_pufferlib_pipeline_do_not_expose_teacher_export_shortcuts(void) {
    static char decoder_header[256000];
    static char decoder_source[256000];
    static char pipeline_source[256000];

    read_file("core/config/decoder_config.h", decoder_header, sizeof(decoder_header));
    read_file("core/config/decoder_config.c", decoder_source, sizeof(decoder_source));
    read_file("scripts/pufferlib_breakout_pipeline.py", pipeline_source, sizeof(pipeline_source));

    require_absent(decoder_header, "ROLLOUT_SCORE");
    require_absent(decoder_header, "rollout_score");
    require_absent(decoder_source, "rollout_score");

    require_absent(pipeline_source, "DEFAULT_WEIGHTS");
    require_absent(pipeline_source, "breakout_weights.bin");
    require_absent(pipeline_source, "pretrained-export");
    require_absent(pipeline_source, "checkpoint-export");
    require_absent(pipeline_source, "train-then-export");
    require_absent(pipeline_source, "load_weights");
    require_absent(pipeline_source, "save_weights");
}

static void test_mask_config_and_snake_masks_do_not_expose_guardrail_strategy_names(void) {
    static char mask_header[256000];
    static char mask_source[256000];
    static char snake_header[256000];
    static char snake_source[256000];
    static char run_config_test[256000];
    static char snake_config[256000];

    read_file("core/config/mask_config.h", mask_header, sizeof(mask_header));
    read_file("core/config/mask_config.c", mask_source, sizeof(mask_source));
    read_file("projects/snake/snake.h", snake_header, sizeof(snake_header));
    read_file("projects/snake/snake.c", snake_source, sizeof(snake_source));
    read_file("tests/test_run_config.c", run_config_test, sizeof(run_config_test));
    read_file("projects/snake/token_policy.ini", snake_config, sizeof(snake_config));

    require_absent(mask_header, "SURVIVAL");
    require_absent(mask_header, "IMMEDIATE");
    require_absent(mask_source, "survival");
    require_absent(mask_source, "immediate");
    require_absent(snake_header, "snake_write_action_mask_survival");
    require_absent(snake_header, "snake_write_action_mask_immediate");
    require_absent(snake_source, "snake_write_action_mask_survival");
    require_absent(snake_source, "snake_write_action_mask_immediate");
    require_absent(run_config_test, "strategy = survival");
    require_absent(snake_config, "strategy = survival");
}

static void test_masked_action_guardrail_library_is_not_supported_surface(void) {
    static char makefile[256000];
    static char benchmark_header[256000];
    static char benchmark_source[256000];
    static char train_header[256000];
    static char train_source[256000];
    static char mlp_header[256000];
    static char mlp_source[256000];
    static char benchmark_test[256000];
    static char train_test[256000];
    static char mlp_test[256000];
    static char ini_test[256000];

    read_file("Makefile", makefile, sizeof(makefile));
    read_file("apps/benchmark.h", benchmark_header, sizeof(benchmark_header));
    read_file("apps/benchmark.c", benchmark_source, sizeof(benchmark_source));
    read_file("core/config/train_config.h", train_header, sizeof(train_header));
    read_file("core/config/train_config.c", train_source, sizeof(train_source));
    read_file("core/model/mlp_window.h", mlp_header, sizeof(mlp_header));
    read_file("core/model/mlp_window.c", mlp_source, sizeof(mlp_source));
    read_file("tests/test_benchmark.c", benchmark_test, sizeof(benchmark_test));
    read_file("tests/test_train_config.c", train_test, sizeof(train_test));
    read_file("tests/test_mlp_window.c", mlp_test, sizeof(mlp_test));
    read_file("tests/test_ini.c", ini_test, sizeof(ini_test));

    require_absent(makefile, "ACTION_MASK");
    require_absent(makefile, "action_mask");
    require_absent(benchmark_header, "masked_ce");
    require_absent(benchmark_source, "masked_ce");
    require_absent(benchmark_source, "valid_mask");
    require_absent(benchmark_source, "tkm_action_mask_argmax");
    require_absent(benchmark_source, "action_mask.h");
    require_absent(benchmark_source, "train_masked_cross_entropy");
    require_absent(train_header, "MASKED_CROSS_ENTROPY");
    require_absent(train_header, "masked_cross_entropy");
    require_absent(train_source, "MASKED_CROSS_ENTROPY");
    require_absent(train_source, "masked_cross_entropy");
    require_absent(mlp_header, "train_masked_cross_entropy");
    require_absent(mlp_source, "train_masked_cross_entropy");
    require_absent(benchmark_test, "masked_ce");
    require_absent(train_test, "masked_cross_entropy");
    require_absent(mlp_test, "masked_cross_entropy");
    require_absent(ini_test, "masked_cross_entropy");

    require_file_absent("core/head/action_mask.h");
    require_file_absent("core/head/action_mask.c");
    require_file_absent("tests/test_action_mask.c");
}

static void test_active_config_paths_do_not_expose_action_scorer_or_action_features(void) {
    static char layer_header[256000];
    static char layer_source[256000];
    static char factory_header[256000];
    static char factory_source[256000];
    static char render_source[256000];
    static char layer_test[256000];
    static char factory_test[256000];
    static char run_config_test[256000];

    read_file("core/config/layer_config.h", layer_header, sizeof(layer_header));
    read_file("core/config/layer_config.c", layer_source, sizeof(layer_source));
    read_file("core/config/layer_factory.h", factory_header, sizeof(factory_header));
    read_file("core/config/layer_factory.c", factory_source, sizeof(factory_source));
    read_file("apps/snake_render.c", render_source, sizeof(render_source));
    read_file("tests/test_layer_config.c", layer_test, sizeof(layer_test));
    read_file("tests/test_layer_factory.c", factory_test, sizeof(factory_test));
    read_file("tests/test_run_config.c", run_config_test, sizeof(run_config_test));

    require_absent(layer_header, "TKM_MODEL_ACTION_SCORER");
    require_absent(layer_header, "TkmActionFeatureKind");
    require_absent(layer_header, "action_features");
    require_absent(layer_source, "action_scorer");
    require_absent(layer_source, "action_features");
    require_absent(factory_header, "action_scorer");
    require_absent(factory_source, "action_scorer");
    require_absent(render_source, "projects/snake/action_scorer.ini");
    require_absent(layer_test, "action_scorer");
    require_absent(layer_test, "action_features");
    require_absent(factory_test, "action_scorer");
    require_absent(run_config_test, "action_scorer");
    require_absent(run_config_test, "action_features");

    require_file_absent("projects/snake/action_scorer.ini");
    require_file_absent("projects/centerline/action_scorer.ini");
    require_file_absent("projects/breakout/action_scorer.ini");
}

static void test_action_scorer_library_is_not_kept_as_supported_shortcut_surface(void) {
    static char makefile[256000];

    read_file("Makefile", makefile, sizeof(makefile));

    require_absent(makefile, "ACTION_SCORER");
    require_absent(makefile, "action_scorer");
    require_file_absent("core/model/action_scorer.h");
    require_file_absent("core/model/action_scorer.c");
    require_file_absent("core/train/action_scorer_trainer.h");
    require_file_absent("core/train/action_scorer_trainer.c");
    require_file_absent("tests/test_action_scorer.c");
    require_file_absent("tests/test_action_scorer_trainer.c");
}

static void test_pufferlib_tokenizers_do_not_select_observation_dims_from_action_labels(void) {
    static const char* config_paths[] = {
        "projects/breakout/pufferlib_token_mlp_first_dims.ini",
        "projects/breakout/pufferlib_token_ngram.ini",
        "projects/breakout/pufferlib_token_backoff_ngram.ini",
        "projects/breakout/pufferlib_token_mlp_window.ini",
        "projects/breakout/pufferlib_token_mlp_window_unmasked.ini",
    };
    static char header[256000];
    static char source[512000];
    static char benchmark[256000];
    static char render[256000];
    static char test_source[512000];
    static char config[64000];

    read_file("projects/breakout/pufferlib_policy.h", header, sizeof(header));
    read_file("projects/breakout/pufferlib_policy.c", source, sizeof(source));
    read_file("apps/pufferlib_breakout_benchmark.c", benchmark, sizeof(benchmark));
    read_file("apps/pufferlib_breakout_policy_render.c", render, sizeof(render));
    read_file("tests/test_breakout_pufferlib_policy.c", test_source, sizeof(test_source));

    require_absent(header, "ACTION_SEPARATION");
    require_absent(header, "TYPED_NEXT_TOKEN");
    require_absent(header, "NGRAM_OBS_CANDIDATES");
    require_absent(header, "obs_counts");
    require_absent(header, "action_counts");
    require_absent(header, "ngram_action_prior");
    require_absent(source, "action_separation");
    require_absent(source, "first_then_action_separation");
    require_absent(source, "best_score");
    require_absent(source, "best_dim");
    require_absent(source, "typed_next_token");
    require_absent(source, "TYPED_NEXT_TOKEN");
    require_absent(source, "breakout_token_mlp_make_mask");
    require_absent(source, "train_masked_cross_entropy");
    require_absent(source, "expected_type");
    require_absent(source, "ngram_update_obs");
    require_absent(source, "ngram_update_action");
    require_absent(source, "ngram_action_prior");
    require_absent(benchmark, "first_then_action_separation");
    require_absent(render, "first_then_action_separation");
    require_absent(test_source, "action_separation");
    require_absent(test_source, "first_then_action_separation");
    require_absent(test_source, "action_mlp_from_token_context");
    require_absent(test_source, "typed_next_token");
    require_absent(test_source, "TYPED_NEXT_TOKEN");

    for (size_t i = 0u; i < sizeof(config_paths) / sizeof(config_paths[0]); i++) {
        read_file(config_paths[i], config, sizeof(config));
        require_absent(config, "action_separation");
        require_absent(config, "first_then_action_separation");
        require_absent(config, "typed_next_token");
    }
}

int main(void) {
    test_active_benchmark_source_has_no_snake_shortcut_symbols();
    test_default_benchmark_source_has_no_hand_coded_action_scorer_surfaces();
    test_default_benchmark_source_has_no_untrained_hard_coded_policy_entries();
    test_default_benchmark_source_has_no_direct_feature_to_action_baselines();
    test_generic_direct_action_policy_libraries_are_removed();
    test_pufferlib_benchmark_source_only_exposes_token_policy_methods();
    test_pufferlib_render_source_only_exposes_token_policy_methods();
    test_pufferlib_smoke_source_evaluates_token_policy_not_direct_action_policy();
    test_pufferlib_data_paths_do_not_use_puffernet_teacher_models();
    test_pufferlib_policy_library_exposes_only_token_policy_path();
    test_snake_sources_do_not_expose_oracle_or_planner_surfaces();
    test_centerline_breakout_and_config_do_not_expose_oracle_surfaces();
    test_decoder_and_pufferlib_pipeline_do_not_expose_teacher_export_shortcuts();
    test_mask_config_and_snake_masks_do_not_expose_guardrail_strategy_names();
    test_masked_action_guardrail_library_is_not_supported_surface();
    test_active_config_paths_do_not_expose_action_scorer_or_action_features();
    test_action_scorer_library_is_not_kept_as_supported_shortcut_surface();
    test_pufferlib_tokenizers_do_not_select_observation_dims_from_action_labels();
    puts("shortcut inventory tests passed");
    return 0;
}
