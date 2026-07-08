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
TkmBenchResult tkm_bench_centerline_sparse_lookup(void);
TkmBenchResult tkm_bench_breakout(void);
TkmBenchResult tkm_bench_breakout_ngram(void);
TkmBenchResult tkm_bench_breakout_sparse_lookup(void);
TkmBenchResult tkm_bench_snake_lookup(void);
TkmBenchResult tkm_bench_snake_ngram(void);
TkmBenchResult tkm_bench_snake_sparse_lookup(void);
TkmBenchResult tkm_bench_snake_sparse_lookup_multistart(void);
void tkm_print_bench_result(TkmBenchResult result);

#endif
