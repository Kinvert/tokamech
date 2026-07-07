#ifndef TKM_BENCHMARK_H
#define TKM_BENCHMARK_H

#include <stdint.h>

typedef struct {
    const char* name;
    uint32_t steps;
    int score;
    uint32_t metric;
} TkmBenchResult;

TkmBenchResult tkm_bench_centerline(void);
TkmBenchResult tkm_bench_centerline_ngram(void);
TkmBenchResult tkm_bench_centerline_mlp(void);
TkmBenchResult tkm_bench_centerline_mlp_trained(void);
TkmBenchResult tkm_bench_centerline_mlp_ce_trained(void);
TkmBenchResult tkm_bench_centerline_mlp_masked_ce_trained(void);
TkmBenchResult tkm_bench_centerline_sparse_lookup(void);
TkmBenchResult tkm_bench_centerline_nearest(void);
TkmBenchResult tkm_bench_centerline_linear_policy(void);
TkmBenchResult tkm_bench_centerline_action_scorer(void);
TkmBenchResult tkm_bench_centerline_action_scorer_config_file(const char* path);
TkmBenchResult tkm_bench_breakout(void);
TkmBenchResult tkm_bench_breakout_ngram(void);
TkmBenchResult tkm_bench_breakout_sparse_lookup(void);
TkmBenchResult tkm_bench_breakout_nearest(void);
TkmBenchResult tkm_bench_breakout_linear_policy(void);
TkmBenchResult tkm_bench_breakout_action_scorer(void);
TkmBenchResult tkm_bench_breakout_action_scorer_config_file(const char* path);
TkmBenchResult tkm_bench_breakout_mlp_ce_trained(void);
TkmBenchResult tkm_bench_breakout_mlp_masked_ce_trained(void);
TkmBenchResult tkm_bench_snake_lookup(void);
TkmBenchResult tkm_bench_snake_ngram(void);
TkmBenchResult tkm_bench_snake_sparse_lookup(void);
TkmBenchResult tkm_bench_snake_sparse_lookup_multistart(void);
TkmBenchResult tkm_bench_snake_nearest_multistart(void);
TkmBenchResult tkm_bench_snake_nearest_safe_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_safe_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_ce_safe_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_ce_masked_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_masked_ce_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_dagger_masked_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_dagger64_masked_holdout(void);
TkmBenchResult tkm_bench_snake_mlp_action_features_holdout(void);
TkmBenchResult tkm_bench_snake_action_scorer_holdout(void);
TkmBenchResult tkm_bench_snake_action_scorer_ce_holdout(void);
TkmBenchResult tkm_bench_snake_action_scorer_config_holdout(const char* train_ini);
TkmBenchResult tkm_bench_snake_action_scorer_config_file_holdout(const char* path);
TkmBenchResult tkm_bench_snake_action_scorer_config_file_long_holdout(const char* path);
TkmBenchResult tkm_bench_snake_planner(void);
void tkm_print_bench_result(TkmBenchResult result);

#endif
