CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -I.
PUFFERLIB_CC ?= clang
PUFFERLIB_MATCH_CFLAGS ?= -O2 -DNDEBUG -mavx2 -mfma

CENTERLINE_TEST_BIN := build/test_centerline_phase0
BREAKOUT_TEST_BIN := build/test_breakout_phase1
SNAKE_TEST_BIN := build/test_snake_phase2
BENCHMARK_TEST_BIN := build/test_benchmark
LOSS_TEST_BIN := build/test_loss
VQ_CODE_TEST_BIN := build/test_vq_code
EMBEDDER_TEST_BIN := build/test_embedder
SEQUENCE_LAYOUT_TEST_BIN := build/test_sequence_layout
INI_TEST_BIN := build/test_ini
LAYER_CONFIG_TEST_BIN := build/test_layer_config
TRAIN_CONFIG_TEST_BIN := build/test_train_config
PROJECT_CONFIG_TEST_BIN := build/test_project_config
MASK_CONFIG_TEST_BIN := build/test_mask_config
DECODER_CONFIG_TEST_BIN := build/test_decoder_config
RUN_CONFIG_TEST_BIN := build/test_run_config
CATEGORICAL_HEAD_TEST_BIN := build/test_categorical_head
ACTION_MASK_TEST_BIN := build/test_action_mask
MLP_WINDOW_TEST_BIN := build/test_mlp_window
ACTION_SCORER_TEST_BIN := build/test_action_scorer
ACTION_SCORER_TRAINER_TEST_BIN := build/test_action_scorer_trainer
NEAREST_POLICY_TEST_BIN := build/test_nearest_policy
LINEAR_POLICY_TEST_BIN := build/test_linear_policy
NGRAM_MODEL_TEST_BIN := build/test_ngram_model
CONTINUOUS_VECTORIZER_TEST_BIN := build/test_continuous_vectorizer
LAYER_FACTORY_TEST_BIN := build/test_layer_factory
PARAMS_TEST_BIN := build/test_params
SPARSE_LOOKUP_TEST_BIN := build/test_sparse_lookup_policy
PUFFERLIB_BREAKOUT_DATASET_TEST_BIN := build/test_pufferlib_breakout_dataset
BREAKOUT_PUFFERLIB_POLICY_TEST_BIN := build/test_breakout_pufferlib_policy
PUFFERLIB_BREAKOUT_TRAIN_EXPORT_HOOK_TEST_BIN := build/test_pufferlib_breakout_train_export_hook
CENTERLINE_RENDER_BIN := build/centerline_render
BREAKOUT_RENDER_BIN := build/breakout_render
SNAKE_RENDER_BIN := build/snake_render
BENCHMARK_BIN := build/benchmark
PUFFERLIB_BREAKOUT_SMOKE_BIN := build/pufferlib_breakout_smoke
PUFFERLIB_BREAKOUT_POLICY_RENDER_BIN := build/pufferlib_breakout_policy_render
PUFFERLIB_BREAKOUT_BENCHMARK_BIN := build/pufferlib_breakout_benchmark
PUFFERLIB_BREAKOUT_CHECKPOINT_EXPORT_BIN := build/pufferlib_breakout_checkpoint_export
PUFFERLIB_DIR ?= /home/claude/pathfinder

CORE_SRC := \
	core/config/ini.c \
	core/config/layer_factory.c \
	core/config/layer_config.c \
	core/config/train_config.c \
	core/config/project_config.c \
	core/config/mask_config.c \
	core/config/decoder_config.c \
	core/config/run_config.c \
	core/dataset/float_transition.c \
	core/dataset/trajectory.c \
	core/embedder/embedder.c \
	core/head/action_mask.c \
	core/head/categorical.c \
	core/loss/loss.c \
	core/model/linear_policy.c \
	core/model/action_scorer.c \
	core/model/mlp_window.c \
	core/model/nearest_policy.c \
	core/model/ngram.c \
	core/params/params.c \
	core/sequence/layout.c \
	core/tokenizer/int_bins.c \
	core/tokenizer/vq_code.c \
	core/train/lookup_policy.c \
	core/train/action_scorer_trainer.c \
	core/train/sparse_lookup_policy.c \
	core/vectorizer/continuous.c \
	core/decoder/discrete_action.c \
	core/runtime/runtime.c

CENTERLINE_SRC := \
	$(CORE_SRC) \
	projects/centerline/centerline.c

BREAKOUT_SRC := \
	$(CORE_SRC) \
	projects/breakout/breakout.c \
	projects/breakout/pufferlib_policy.c

SNAKE_SRC := \
	$(CORE_SRC) \
	projects/snake/snake.c

PROJECT_SRC := \
	$(CORE_SRC) \
	projects/centerline/centerline.c \
	projects/breakout/breakout.c \
	projects/snake/snake.c

RAYLIB_DIR ?= $(firstword $(wildcard raylib-5.5_linux_amd64 /home/claude/pathfinder/raylib-5.5_linux_amd64))
RAYLIB_CFLAGS ?= -I$(RAYLIB_DIR)/include -DPLATFORM_DESKTOP
RAYLIB_LIBS ?= $(RAYLIB_DIR)/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11
LDLIBS ?= -lm

.PHONY: test benchmark pufferlib-breakout-smoke pufferlib-breakout-policy-render pufferlib-breakout-benchmark pufferlib-breakout-checkpoint-export centerline-render breakout-render snake-render clean

test: $(CENTERLINE_TEST_BIN) $(BREAKOUT_TEST_BIN) $(SNAKE_TEST_BIN) $(BENCHMARK_TEST_BIN) $(LOSS_TEST_BIN) $(VQ_CODE_TEST_BIN) $(EMBEDDER_TEST_BIN) $(SEQUENCE_LAYOUT_TEST_BIN) $(INI_TEST_BIN) $(LAYER_CONFIG_TEST_BIN) $(TRAIN_CONFIG_TEST_BIN) $(PROJECT_CONFIG_TEST_BIN) $(MASK_CONFIG_TEST_BIN) $(DECODER_CONFIG_TEST_BIN) $(RUN_CONFIG_TEST_BIN) $(CATEGORICAL_HEAD_TEST_BIN) $(ACTION_MASK_TEST_BIN) $(MLP_WINDOW_TEST_BIN) $(ACTION_SCORER_TEST_BIN) $(ACTION_SCORER_TRAINER_TEST_BIN) $(NEAREST_POLICY_TEST_BIN) $(LINEAR_POLICY_TEST_BIN) $(NGRAM_MODEL_TEST_BIN) $(CONTINUOUS_VECTORIZER_TEST_BIN) $(LAYER_FACTORY_TEST_BIN) $(PARAMS_TEST_BIN) $(SPARSE_LOOKUP_TEST_BIN) $(PUFFERLIB_BREAKOUT_DATASET_TEST_BIN) $(BREAKOUT_PUFFERLIB_POLICY_TEST_BIN) $(PUFFERLIB_BREAKOUT_TRAIN_EXPORT_HOOK_TEST_BIN)
	./$(CENTERLINE_TEST_BIN)
	./$(BREAKOUT_TEST_BIN)
	./$(SNAKE_TEST_BIN)
	./$(BENCHMARK_TEST_BIN)
	./$(LOSS_TEST_BIN)
	./$(VQ_CODE_TEST_BIN)
	./$(EMBEDDER_TEST_BIN)
	./$(SEQUENCE_LAYOUT_TEST_BIN)
	./$(INI_TEST_BIN)
	./$(LAYER_CONFIG_TEST_BIN)
	./$(TRAIN_CONFIG_TEST_BIN)
	./$(PROJECT_CONFIG_TEST_BIN)
	./$(MASK_CONFIG_TEST_BIN)
	./$(DECODER_CONFIG_TEST_BIN)
	./$(RUN_CONFIG_TEST_BIN)
	./$(CATEGORICAL_HEAD_TEST_BIN)
	./$(ACTION_MASK_TEST_BIN)
	./$(MLP_WINDOW_TEST_BIN)
	./$(ACTION_SCORER_TEST_BIN)
	./$(ACTION_SCORER_TRAINER_TEST_BIN)
	./$(NEAREST_POLICY_TEST_BIN)
	./$(LINEAR_POLICY_TEST_BIN)
	./$(NGRAM_MODEL_TEST_BIN)
	./$(CONTINUOUS_VECTORIZER_TEST_BIN)
	./$(LAYER_FACTORY_TEST_BIN)
	./$(PARAMS_TEST_BIN)
	./$(SPARSE_LOOKUP_TEST_BIN)
	./$(PUFFERLIB_BREAKOUT_DATASET_TEST_BIN)
	./$(BREAKOUT_PUFFERLIB_POLICY_TEST_BIN)
	./$(PUFFERLIB_BREAKOUT_TRAIN_EXPORT_HOOK_TEST_BIN)

benchmark: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN)

pufferlib-breakout-smoke: $(PUFFERLIB_BREAKOUT_SMOKE_BIN)
	./$(PUFFERLIB_BREAKOUT_SMOKE_BIN)

pufferlib-breakout-policy-render: $(PUFFERLIB_BREAKOUT_POLICY_RENDER_BIN)
	./$(PUFFERLIB_BREAKOUT_POLICY_RENDER_BIN)

pufferlib-breakout-benchmark: $(PUFFERLIB_BREAKOUT_BENCHMARK_BIN)
	./$(PUFFERLIB_BREAKOUT_BENCHMARK_BIN)

pufferlib-breakout-checkpoint-export: $(PUFFERLIB_BREAKOUT_CHECKPOINT_EXPORT_BIN)
	./$(PUFFERLIB_BREAKOUT_CHECKPOINT_EXPORT_BIN)

centerline-render: $(CENTERLINE_RENDER_BIN)
	./$(CENTERLINE_RENDER_BIN)

breakout-render: $(BREAKOUT_RENDER_BIN)
	./$(BREAKOUT_RENDER_BIN)

snake-render: $(SNAKE_RENDER_BIN)
	./$(SNAKE_RENDER_BIN)

$(CENTERLINE_TEST_BIN): tests/test_centerline_phase0.c $(CENTERLINE_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BREAKOUT_TEST_BIN): tests/test_breakout_phase1.c $(BREAKOUT_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(SNAKE_TEST_BIN): tests/test_snake_phase2.c $(SNAKE_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BENCHMARK_TEST_BIN): tests/test_benchmark.c apps/benchmark.c $(PROJECT_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) -DTKM_BENCHMARK_NO_MAIN tests/test_benchmark.c apps/benchmark.c $(PROJECT_SRC) $(LDLIBS) -o $@

$(LOSS_TEST_BIN): tests/test_loss.c core/loss/loss.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_loss.c core/loss/loss.c $(LDLIBS) -o $@

$(VQ_CODE_TEST_BIN): tests/test_vq_code.c core/tokenizer/vq_code.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_vq_code.c core/tokenizer/vq_code.c $(LDLIBS) -o $@

$(EMBEDDER_TEST_BIN): tests/test_embedder.c core/embedder/embedder.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_embedder.c core/embedder/embedder.c $(LDLIBS) -o $@

$(SEQUENCE_LAYOUT_TEST_BIN): tests/test_sequence_layout.c core/sequence/layout.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_sequence_layout.c core/sequence/layout.c $(LDLIBS) -o $@

$(INI_TEST_BIN): tests/test_ini.c core/config/ini.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_ini.c core/config/ini.c $(LDLIBS) -o $@

$(LAYER_CONFIG_TEST_BIN): tests/test_layer_config.c core/config/ini.c core/config/layer_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_layer_config.c core/config/ini.c core/config/layer_config.c $(LDLIBS) -o $@

$(TRAIN_CONFIG_TEST_BIN): tests/test_train_config.c core/config/ini.c core/config/train_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_train_config.c core/config/ini.c core/config/train_config.c $(LDLIBS) -o $@

$(PROJECT_CONFIG_TEST_BIN): tests/test_project_config.c core/config/ini.c core/config/project_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_project_config.c core/config/ini.c core/config/project_config.c $(LDLIBS) -o $@

$(MASK_CONFIG_TEST_BIN): tests/test_mask_config.c core/config/ini.c core/config/mask_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_mask_config.c core/config/ini.c core/config/mask_config.c $(LDLIBS) -o $@

$(DECODER_CONFIG_TEST_BIN): tests/test_decoder_config.c core/config/ini.c core/config/decoder_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_decoder_config.c core/config/ini.c core/config/decoder_config.c $(LDLIBS) -o $@

$(RUN_CONFIG_TEST_BIN): tests/test_run_config.c core/config/ini.c core/config/project_config.c core/config/layer_config.c core/config/train_config.c core/config/mask_config.c core/config/decoder_config.c core/config/run_config.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_run_config.c core/config/ini.c core/config/project_config.c core/config/layer_config.c core/config/train_config.c core/config/mask_config.c core/config/decoder_config.c core/config/run_config.c $(LDLIBS) -o $@

$(CATEGORICAL_HEAD_TEST_BIN): tests/test_categorical_head.c core/head/categorical.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_categorical_head.c core/head/categorical.c $(LDLIBS) -o $@

$(ACTION_MASK_TEST_BIN): tests/test_action_mask.c core/head/action_mask.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_action_mask.c core/head/action_mask.c $(LDLIBS) -o $@

$(MLP_WINDOW_TEST_BIN): tests/test_mlp_window.c core/model/mlp_window.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_mlp_window.c core/model/mlp_window.c $(LDLIBS) -o $@

$(ACTION_SCORER_TEST_BIN): tests/test_action_scorer.c core/model/action_scorer.c core/model/mlp_window.c core/head/action_mask.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_action_scorer.c core/model/action_scorer.c core/model/mlp_window.c core/head/action_mask.c $(LDLIBS) -o $@

$(ACTION_SCORER_TRAINER_TEST_BIN): tests/test_action_scorer_trainer.c core/train/action_scorer_trainer.c core/model/action_scorer.c core/model/mlp_window.c core/head/action_mask.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_action_scorer_trainer.c core/train/action_scorer_trainer.c core/model/action_scorer.c core/model/mlp_window.c core/head/action_mask.c $(LDLIBS) -o $@

$(NEAREST_POLICY_TEST_BIN): tests/test_nearest_policy.c core/model/nearest_policy.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_nearest_policy.c core/model/nearest_policy.c $(LDLIBS) -o $@

$(LINEAR_POLICY_TEST_BIN): tests/test_linear_policy.c core/model/linear_policy.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_linear_policy.c core/model/linear_policy.c $(LDLIBS) -o $@

$(NGRAM_MODEL_TEST_BIN): tests/test_ngram_model.c core/model/ngram.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_ngram_model.c core/model/ngram.c $(LDLIBS) -o $@

$(CONTINUOUS_VECTORIZER_TEST_BIN): tests/test_continuous_vectorizer.c core/vectorizer/continuous.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_continuous_vectorizer.c core/vectorizer/continuous.c $(LDLIBS) -o $@

$(LAYER_FACTORY_TEST_BIN): tests/test_layer_factory.c core/config/ini.c core/config/layer_config.c core/config/layer_factory.c core/embedder/embedder.c core/tokenizer/int_bins.c core/tokenizer/vq_code.c core/vectorizer/continuous.c core/model/mlp_window.c core/model/action_scorer.c core/model/ngram.c core/head/action_mask.c core/params/params.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_layer_factory.c core/config/ini.c core/config/layer_config.c core/config/layer_factory.c core/embedder/embedder.c core/tokenizer/int_bins.c core/tokenizer/vq_code.c core/vectorizer/continuous.c core/model/mlp_window.c core/model/action_scorer.c core/model/ngram.c core/head/action_mask.c core/params/params.c $(LDLIBS) -o $@

$(PARAMS_TEST_BIN): tests/test_params.c core/params/params.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_params.c core/params/params.c $(LDLIBS) -o $@

$(SPARSE_LOOKUP_TEST_BIN): tests/test_sparse_lookup_policy.c core/train/sparse_lookup_policy.c core/dataset/trajectory.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_sparse_lookup_policy.c core/train/sparse_lookup_policy.c core/dataset/trajectory.c $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_DATASET_TEST_BIN): tests/test_pufferlib_breakout_dataset.c core/dataset/float_transition.c core/dataset/float_transition.h
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_pufferlib_breakout_dataset.c core/dataset/float_transition.c $(LDLIBS) -o $@

$(BREAKOUT_PUFFERLIB_POLICY_TEST_BIN): tests/test_breakout_pufferlib_policy.c projects/breakout/pufferlib_policy.c projects/breakout/pufferlib_policy.h core/dataset/float_transition.c core/dataset/float_transition.h core/model/mlp_window.c core/config/ini.c
	mkdir -p build
	$(CC) $(CFLAGS) tests/test_breakout_pufferlib_policy.c projects/breakout/pufferlib_policy.c core/dataset/float_transition.c core/model/mlp_window.c core/config/ini.c $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_TRAIN_EXPORT_HOOK_TEST_BIN): tests/test_pufferlib_breakout_train_export_hook.c
	mkdir -p build
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/binding.c"
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/exporter.h"
	test -n "$(RAYLIB_DIR)"
	$(PUFFERLIB_CC) -std=gnu11 -Wall -Wextra -Werror=return-type -Wno-unused-function -Wno-unused-parameter -I$(PUFFERLIB_DIR)/ocean/breakout -I$(PUFFERLIB_DIR)/src -I$(PUFFERLIB_DIR)/vendor $(RAYLIB_CFLAGS) tests/test_pufferlib_breakout_train_export_hook.c $(RAYLIB_LIBS) -fopenmp -o $@

$(BENCHMARK_BIN): apps/benchmark.c $(PROJECT_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) apps/benchmark.c $(PROJECT_SRC) $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_SMOKE_BIN): apps/pufferlib_breakout_smoke.c projects/breakout/pufferlib_policy.c projects/breakout/pufferlib_policy.h core/dataset/float_transition.c core/dataset/float_transition.h core/model/mlp_window.c core/config/ini.c
	mkdir -p build
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/breakout.h"
	test -f "$(PUFFERLIB_DIR)/src/puffernet.h"
	test -f "$(PUFFERLIB_DIR)/resources/breakout/breakout_weights.bin"
	test -n "$(RAYLIB_DIR)"
	$(PUFFERLIB_CC) -I$(PUFFERLIB_DIR)/ocean/breakout -I$(PUFFERLIB_DIR)/src $(filter-out -Werror -pedantic,$(CFLAGS)) $(PUFFERLIB_MATCH_CFLAGS) -Wno-unused-parameter $(RAYLIB_CFLAGS) -D_DEFAULT_SOURCE -DPUFFERLIB_DIR=\"$(PUFFERLIB_DIR)\" apps/pufferlib_breakout_smoke.c projects/breakout/pufferlib_policy.c core/dataset/float_transition.c core/model/mlp_window.c core/config/ini.c $(RAYLIB_LIBS) $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_POLICY_RENDER_BIN): apps/pufferlib_breakout_policy_render.c projects/breakout/pufferlib_policy.c projects/breakout/pufferlib_policy.h core/dataset/float_transition.c core/dataset/float_transition.h core/model/mlp_window.c core/config/ini.c
	mkdir -p build
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/breakout.h"
	test -f "$(PUFFERLIB_DIR)/resources/shared/puffers_128.png"
	test -n "$(RAYLIB_DIR)"
	$(PUFFERLIB_CC) -I$(PUFFERLIB_DIR)/ocean/breakout $(filter-out -Werror -pedantic,$(CFLAGS)) $(PUFFERLIB_MATCH_CFLAGS) -Wno-unused-parameter $(RAYLIB_CFLAGS) -D_DEFAULT_SOURCE -DPUFFERLIB_DIR=\"$(PUFFERLIB_DIR)\" apps/pufferlib_breakout_policy_render.c projects/breakout/pufferlib_policy.c core/dataset/float_transition.c core/model/mlp_window.c core/config/ini.c $(RAYLIB_LIBS) $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_BENCHMARK_BIN): apps/pufferlib_breakout_benchmark.c projects/breakout/pufferlib_policy.c projects/breakout/pufferlib_policy.h core/dataset/float_transition.c core/dataset/float_transition.h core/model/mlp_window.c core/config/ini.c
	mkdir -p build
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/breakout.h"
	test -n "$(RAYLIB_DIR)"
	$(PUFFERLIB_CC) -I$(PUFFERLIB_DIR)/ocean/breakout $(filter-out -Werror -pedantic,$(CFLAGS)) $(PUFFERLIB_MATCH_CFLAGS) -Wno-unused-parameter $(RAYLIB_CFLAGS) -D_DEFAULT_SOURCE -DPUFFERLIB_DIR=\"$(PUFFERLIB_DIR)\" apps/pufferlib_breakout_benchmark.c projects/breakout/pufferlib_policy.c core/dataset/float_transition.c core/model/mlp_window.c core/config/ini.c $(RAYLIB_LIBS) $(LDLIBS) -o $@

$(PUFFERLIB_BREAKOUT_CHECKPOINT_EXPORT_BIN): apps/pufferlib_breakout_checkpoint_export.c
	mkdir -p build
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/breakout.h"
	test -f "$(PUFFERLIB_DIR)/ocean/breakout/exporter.h"
	test -f "$(PUFFERLIB_DIR)/src/puffernet.h"
	test -n "$(RAYLIB_DIR)"
	$(PUFFERLIB_CC) -I$(PUFFERLIB_DIR)/ocean/breakout -I$(PUFFERLIB_DIR)/src $(filter-out -Werror -pedantic,$(CFLAGS)) $(PUFFERLIB_MATCH_CFLAGS) -Wno-unused-parameter $(RAYLIB_CFLAGS) -D_DEFAULT_SOURCE apps/pufferlib_breakout_checkpoint_export.c $(RAYLIB_LIBS) $(LDLIBS) -o $@

$(CENTERLINE_RENDER_BIN): apps/centerline_render.c $(CENTERLINE_SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

$(BREAKOUT_RENDER_BIN): apps/breakout_render.c $(BREAKOUT_SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

$(SNAKE_RENDER_BIN): apps/snake_render.c $(SNAKE_SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

clean:
	rm -rf build
