#include <stdio.h>
#include <stdlib.h>

#include "apps/benchmark.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

extern TkmBenchResult tkm_bench_snake_planner(void);
extern TkmBenchResult tkm_bench_snake_mlp_safe_holdout(void);

static void test_project_benchmarks_return_sane_metrics(void) {
    TkmBenchResult centerline = tkm_bench_centerline();
    TkmBenchResult centerline_ngram = tkm_bench_centerline_ngram();
    TkmBenchResult centerline_mlp = tkm_bench_centerline_mlp();
    TkmBenchResult centerline_mlp_trained = tkm_bench_centerline_mlp_trained();
    TkmBenchResult centerline_mlp_ce = tkm_bench_centerline_mlp_ce_trained();
    TkmBenchResult centerline_mlp_masked_ce = tkm_bench_centerline_mlp_masked_ce_trained();
    TkmBenchResult centerline_sparse = tkm_bench_centerline_sparse_lookup();
    TkmBenchResult centerline_nearest = tkm_bench_centerline_nearest();
    TkmBenchResult centerline_linear = tkm_bench_centerline_linear_policy();
    TkmBenchResult centerline_action_scorer = tkm_bench_centerline_action_scorer();
    TkmBenchResult centerline_action_scorer_config_file =
        tkm_bench_centerline_action_scorer_config_file("projects/centerline/action_scorer.ini");
    TkmBenchResult centerline_action_scorer_config_missing =
        tkm_bench_centerline_action_scorer_config_file("projects/centerline/missing.ini");
    TkmBenchResult centerline_action_scorer_config_wrong_project =
        tkm_bench_centerline_action_scorer_config_file("projects/breakout/action_scorer.ini");
    TkmBenchResult breakout = tkm_bench_breakout();
    TkmBenchResult breakout_ngram = tkm_bench_breakout_ngram();
    TkmBenchResult breakout_sparse = tkm_bench_breakout_sparse_lookup();
    TkmBenchResult breakout_nearest = tkm_bench_breakout_nearest();
    TkmBenchResult breakout_linear = tkm_bench_breakout_linear_policy();
    TkmBenchResult breakout_action_scorer = tkm_bench_breakout_action_scorer();
    TkmBenchResult breakout_action_scorer_config_file =
        tkm_bench_breakout_action_scorer_config_file("projects/breakout/action_scorer.ini");
    TkmBenchResult breakout_action_scorer_config_missing =
        tkm_bench_breakout_action_scorer_config_file("projects/breakout/missing.ini");
    TkmBenchResult breakout_action_scorer_config_wrong_project =
        tkm_bench_breakout_action_scorer_config_file("projects/centerline/action_scorer.ini");
    TkmBenchResult breakout_mlp_ce = tkm_bench_breakout_mlp_ce_trained();
    TkmBenchResult breakout_mlp_masked_ce = tkm_bench_breakout_mlp_masked_ce_trained();
    TkmBenchResult snake = tkm_bench_snake_lookup();
    TkmBenchResult snake_ngram = tkm_bench_snake_ngram();
    TkmBenchResult snake_sparse = tkm_bench_snake_sparse_lookup();
    TkmBenchResult snake_sparse_multi = tkm_bench_snake_sparse_lookup_multistart();
    TkmBenchResult snake_nearest_multi = tkm_bench_snake_nearest_multistart();
    TkmBenchResult snake_nearest_safe = tkm_bench_snake_nearest_safe_holdout();
    TkmBenchResult snake_mlp_safe = tkm_bench_snake_mlp_safe_holdout();
    TkmBenchResult snake_mlp_ce_safe = tkm_bench_snake_mlp_ce_safe_holdout();
    TkmBenchResult snake_mlp_ce_masked = tkm_bench_snake_mlp_ce_masked_holdout();
    TkmBenchResult snake_mlp_masked_ce = tkm_bench_snake_mlp_masked_ce_holdout();
    TkmBenchResult snake_mlp_dagger_masked = tkm_bench_snake_mlp_dagger_masked_holdout();
    TkmBenchResult snake_mlp_dagger64_masked = tkm_bench_snake_mlp_dagger64_masked_holdout();
    TkmBenchResult snake_mlp_action_features = tkm_bench_snake_mlp_action_features_holdout();
    TkmBenchResult snake_action_scorer = tkm_bench_snake_action_scorer_holdout();
    TkmBenchResult snake_action_scorer_ce = tkm_bench_snake_action_scorer_ce_holdout();
    TkmBenchResult snake_action_scorer_config_margin =
        tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = margin\n[mask]\nstrategy = survival\n");
    TkmBenchResult snake_action_scorer_config_ce =
        tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = masked_cross_entropy\n");
    TkmBenchResult snake_action_scorer_config_survival =
        tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = masked_cross_entropy\n[mask]\nstrategy = survival\n");
    TkmBenchResult snake_action_scorer_config_bad =
        tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = action_scorer\n[train]\nloss = unknown\n");
    TkmBenchResult snake_action_scorer_config_wrong_model =
        tkm_bench_snake_action_scorer_config_holdout("[project]\nname = snake\n[model]\nkind = lookup_policy\n[train]\nloss = masked_cross_entropy\n");
    TkmBenchResult snake_action_scorer_config_file =
        tkm_bench_snake_action_scorer_config_file_holdout("projects/snake/action_scorer.ini");
    TkmBenchResult snake_action_scorer_config_file_long =
        tkm_bench_snake_action_scorer_config_file_long_holdout("projects/snake/action_scorer.ini");
    TkmBenchResult snake_action_scorer_config_missing =
        tkm_bench_snake_action_scorer_config_file_holdout("projects/snake/missing.ini");
    TkmBenchResult snake_planner = tkm_bench_snake_planner();

    CHECK(centerline.steps > 0);
    CHECK(centerline.score >= 0);
    CHECK(centerline.metric > 0);
    CHECK(centerline_ngram.steps > 0);
    CHECK(centerline_ngram.score >= 0);
    CHECK(centerline_ngram.metric > 0);
    CHECK(centerline_mlp.steps > 0);
    CHECK(centerline_mlp.score >= 0);
    CHECK(centerline_mlp.metric > 0);
    CHECK(centerline_mlp_trained.steps > 0);
    CHECK(centerline_mlp_trained.score >= 0);
    CHECK(centerline_mlp_trained.metric > 0);
    CHECK(centerline_mlp_ce.steps > 0);
    CHECK(centerline_mlp_ce.score >= 0);
    CHECK(centerline_mlp_ce.metric > 0);
    CHECK(centerline_mlp_masked_ce.steps > 0);
    CHECK(centerline_mlp_masked_ce.score >= 0);
    CHECK(centerline_mlp_masked_ce.metric > 0);
    CHECK(centerline_sparse.steps > 0);
    CHECK(centerline_sparse.score >= 0);
    CHECK(centerline_sparse.metric > 0);
    CHECK(centerline_nearest.steps > 0);
    CHECK(centerline_nearest.score >= 0);
    CHECK(centerline_nearest.metric > 0);
    CHECK(centerline_linear.steps > 0);
    CHECK(centerline_linear.score >= 0);
    CHECK(centerline_linear.metric > 0);
    CHECK(centerline_action_scorer.steps > 0);
    CHECK(centerline_action_scorer.score >= 0);
    CHECK(centerline_action_scorer.metric > 0);
    CHECK(centerline_action_scorer_config_file.steps > 0);
    CHECK(centerline_action_scorer_config_file.score >= 0);
    CHECK(centerline_action_scorer_config_file.metric > 0);
    CHECK(centerline_action_scorer_config_missing.steps == 0);
    CHECK(centerline_action_scorer_config_missing.score == 0);
    CHECK(centerline_action_scorer_config_missing.metric == 0);
    CHECK(centerline_action_scorer_config_wrong_project.steps == 0);
    CHECK(centerline_action_scorer_config_wrong_project.score == 0);
    CHECK(centerline_action_scorer_config_wrong_project.metric == 0);
    CHECK(breakout.steps > 0);
    CHECK(breakout.score >= 0);
    CHECK(breakout.metric > 0);
    CHECK(breakout_ngram.steps > 0);
    CHECK(breakout_ngram.score >= 0);
    CHECK(breakout_ngram.metric > 0);
    CHECK(breakout_sparse.steps > 0);
    CHECK(breakout_sparse.score >= 0);
    CHECK(breakout_sparse.metric >= breakout.metric);
    CHECK(breakout_nearest.steps > 0);
    CHECK(breakout_nearest.score >= 0);
    CHECK(breakout_nearest.metric >= breakout.metric);
    CHECK(breakout_linear.steps > 0);
    CHECK(breakout_linear.score >= 0);
    CHECK(breakout_linear.metric >= breakout.metric);
    CHECK(breakout_action_scorer.steps > 0);
    CHECK(breakout_action_scorer.score >= 0);
    CHECK(breakout_action_scorer.metric >= breakout.metric);
    CHECK(breakout_action_scorer_config_file.steps > 0);
    CHECK(breakout_action_scorer_config_file.score >= 0);
    CHECK(breakout_action_scorer_config_file.metric >= breakout.metric);
    CHECK(breakout_action_scorer_config_missing.steps == 0);
    CHECK(breakout_action_scorer_config_missing.score == 0);
    CHECK(breakout_action_scorer_config_missing.metric == 0);
    CHECK(breakout_action_scorer_config_wrong_project.steps == 0);
    CHECK(breakout_action_scorer_config_wrong_project.score == 0);
    CHECK(breakout_action_scorer_config_wrong_project.metric == 0);
    CHECK(breakout_mlp_ce.steps > 0);
    CHECK(breakout_mlp_ce.score >= 0);
    CHECK(breakout_mlp_ce.metric >= breakout.metric);
    CHECK(breakout_mlp_masked_ce.steps > 0);
    CHECK(breakout_mlp_masked_ce.score >= 0);
    CHECK(breakout_mlp_masked_ce.metric >= breakout.metric);
    CHECK(snake.steps > 0);
    CHECK(snake.score >= 0);
    CHECK(snake.metric > 0);
    CHECK(snake_ngram.steps > 0);
    CHECK(snake_ngram.score >= 0);
    CHECK(snake_ngram.metric > 0);
    CHECK(snake_sparse.steps >= 90);
    CHECK(snake_sparse.score >= 8);
    CHECK(snake_sparse.metric >= 8);
    CHECK(snake_sparse.metric >= snake.metric);
    CHECK(snake_sparse_multi.steps >= 270);
    CHECK(snake_sparse_multi.score >= 24);
    CHECK(snake_sparse_multi.metric >= 24);
    CHECK(snake_sparse_multi.metric >= snake_sparse.metric);
    CHECK(snake_nearest_multi.steps >= 100);
    CHECK(snake_nearest_multi.score >= 4);
    CHECK(snake_nearest_multi.metric >= 4);
    CHECK(snake_nearest_safe.steps >= 480);
    CHECK(snake_nearest_safe.score >= 40);
    CHECK(snake_nearest_safe.metric >= 40);
    CHECK(snake_mlp_safe.steps >= 480);
    CHECK(snake_mlp_safe.score >= 20);
    CHECK(snake_mlp_safe.metric >= 20);
    CHECK(snake_mlp_ce_safe.steps >= 480);
    CHECK(snake_mlp_ce_safe.score >= 20);
    CHECK(snake_mlp_ce_safe.metric >= 20);
    CHECK(snake_mlp_ce_masked.steps >= 480);
    CHECK(snake_mlp_ce_masked.score >= 20);
    CHECK(snake_mlp_ce_masked.metric >= 20);
    CHECK(snake_mlp_masked_ce.steps >= 400);
    CHECK(snake_mlp_masked_ce.score >= 15);
    CHECK(snake_mlp_masked_ce.metric >= 15);
    CHECK(snake_mlp_dagger_masked.steps >= 480);
    CHECK(snake_mlp_dagger_masked.score >= 35);
    CHECK(snake_mlp_dagger_masked.metric >= 35);
    CHECK(snake_mlp_dagger64_masked.steps >= 480);
    CHECK(snake_mlp_dagger64_masked.score >= 28);
    CHECK(snake_mlp_dagger64_masked.metric >= 28);
    CHECK(snake_mlp_action_features.steps >= 480);
    CHECK(snake_mlp_action_features.score >= 42);
    CHECK(snake_mlp_action_features.metric >= 42);
    CHECK(snake_action_scorer.steps >= 480);
    CHECK(snake_action_scorer.score >= 45);
    CHECK(snake_action_scorer.metric >= 45);
    CHECK(snake_action_scorer_ce.steps >= 400);
    CHECK(snake_action_scorer_ce.score >= 20);
    CHECK(snake_action_scorer_ce.metric >= 20);
    CHECK(snake_action_scorer_config_margin.steps >= 480);
    CHECK(snake_action_scorer_config_margin.score >= 45);
    CHECK(snake_action_scorer_config_margin.metric >= 45);
    CHECK(snake_action_scorer_config_ce.steps >= 480);
    CHECK(snake_action_scorer_config_ce.score >= 45);
    CHECK(snake_action_scorer_config_ce.metric >= 45);
    CHECK(snake_action_scorer_config_survival.steps >= 480);
    CHECK(snake_action_scorer_config_survival.score >= 58);
    CHECK(snake_action_scorer_config_survival.metric >= 58);
    CHECK(snake_action_scorer_config_survival.metric > snake_action_scorer_config_ce.metric);
    CHECK(snake_action_scorer_config_bad.steps == 0);
    CHECK(snake_action_scorer_config_bad.score == 0);
    CHECK(snake_action_scorer_config_bad.metric == 0);
    CHECK(snake_action_scorer_config_wrong_model.steps == 0);
    CHECK(snake_action_scorer_config_wrong_model.score == 0);
    CHECK(snake_action_scorer_config_wrong_model.metric == 0);
    CHECK(snake_action_scorer_config_file.steps >= 480);
    CHECK(snake_action_scorer_config_file.score >= 125);
    CHECK(snake_action_scorer_config_file.metric >= 125);
    CHECK(snake_action_scorer_config_file_long.steps >= 1200);
    CHECK(snake_action_scorer_config_file_long.score >= 245);
    CHECK(snake_action_scorer_config_file_long.metric >= 245);
    CHECK(snake_action_scorer_config_missing.steps == 0);
    CHECK(snake_action_scorer_config_missing.score == 0);
    CHECK(snake_action_scorer_config_missing.metric == 0);
    CHECK(snake_planner.steps > snake.steps);
    CHECK(snake_planner.score >= 4);
    CHECK(snake_planner.metric >= 4);
    CHECK(snake_planner.metric >= snake.metric);
}

int main(void) {
    test_project_benchmarks_return_sane_metrics();
    puts("benchmark tests passed");
    return 0;
}
