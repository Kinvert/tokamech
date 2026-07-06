CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -I.

CENTERLINE_TEST_BIN := build/test_centerline_phase0
BREAKOUT_TEST_BIN := build/test_breakout_phase1
CENTERLINE_RENDER_BIN := build/centerline_render
BREAKOUT_RENDER_BIN := build/breakout_render

CORE_SRC := \
	core/dataset/trajectory.c \
	core/tokenizer/int_bins.c \
	core/train/lookup_policy.c \
	core/decoder/discrete_action.c \
	core/runtime/runtime.c

CENTERLINE_SRC := \
	$(CORE_SRC) \
	projects/centerline/centerline.c

BREAKOUT_SRC := \
	$(CORE_SRC) \
	projects/breakout/breakout.c

RAYLIB_DIR ?= $(firstword $(wildcard raylib-5.5_linux_amd64 /home/claude/pathfinder/raylib-5.5_linux_amd64))
RAYLIB_CFLAGS ?= -I$(RAYLIB_DIR)/include -DPLATFORM_DESKTOP
RAYLIB_LIBS ?= $(RAYLIB_DIR)/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11
LDLIBS ?= -lm

.PHONY: test centerline-render breakout-render clean

test: $(CENTERLINE_TEST_BIN) $(BREAKOUT_TEST_BIN)
	./$(CENTERLINE_TEST_BIN)
	./$(BREAKOUT_TEST_BIN)

centerline-render: $(CENTERLINE_RENDER_BIN)
	./$(CENTERLINE_RENDER_BIN)

breakout-render: $(BREAKOUT_RENDER_BIN)
	./$(BREAKOUT_RENDER_BIN)

$(CENTERLINE_TEST_BIN): tests/test_centerline_phase0.c $(CENTERLINE_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BREAKOUT_TEST_BIN): tests/test_breakout_phase1.c $(BREAKOUT_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(CENTERLINE_RENDER_BIN): apps/centerline_render.c $(CENTERLINE_SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

$(BREAKOUT_RENDER_BIN): apps/breakout_render.c $(BREAKOUT_SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

clean:
	rm -rf build
