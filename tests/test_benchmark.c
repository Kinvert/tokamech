#include <stdio.h>
#include <stdlib.h>

#include "apps/benchmark.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

#define CHECK_BENCH_SANE(result) do { \
    CHECK((result).steps > 0); \
    CHECK((result).score >= 0); \
    CHECK((result).metric <= (result).steps); \
} while (0)

static void test_project_benchmarks_return_sane_metrics(void) {
    TkmBenchResult centerline = tkm_bench_centerline();
    TkmBenchResult centerline_ngram = tkm_bench_centerline_ngram();
    TkmBenchResult centerline_sparse = tkm_bench_centerline_sparse_lookup();
    TkmBenchResult breakout = tkm_bench_breakout();
    TkmBenchResult breakout_ngram = tkm_bench_breakout_ngram();
    TkmBenchResult breakout_sparse = tkm_bench_breakout_sparse_lookup();
    TkmBenchResult snake = tkm_bench_snake_lookup();
    TkmBenchResult snake_ngram = tkm_bench_snake_ngram();
    TkmBenchResult snake_sparse = tkm_bench_snake_sparse_lookup();
    TkmBenchResult snake_sparse_multi = tkm_bench_snake_sparse_lookup_multistart();

    CHECK_BENCH_SANE(centerline);
    CHECK_BENCH_SANE(centerline_ngram);
    CHECK_BENCH_SANE(centerline_sparse);
    CHECK_BENCH_SANE(breakout);
    CHECK_BENCH_SANE(breakout_ngram);
    CHECK_BENCH_SANE(breakout_sparse);
    CHECK_BENCH_SANE(snake);
    CHECK_BENCH_SANE(snake_ngram);
    CHECK_BENCH_SANE(snake_sparse);
    CHECK_BENCH_SANE(snake_sparse_multi);
    CHECK(centerline.metric == 1u);
    CHECK(centerline_ngram.metric == 1u);
    CHECK(centerline_sparse.metric == 1u);
    CHECK(breakout.metric >= 1u);
    CHECK(breakout_ngram.metric >= 1u);
    CHECK(breakout_sparse.metric >= 1u);
    CHECK(snake.metric >= 1u);
    CHECK(snake_ngram.metric >= 1u);
}

int main(void) {
    test_project_benchmarks_return_sane_metrics();
    puts("benchmark tests passed");
    return 0;
}
