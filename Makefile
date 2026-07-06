CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -pedantic -I.

TEST_BIN := build/test_centerline_phase0
CENTERLINE_RENDER_BIN := build/centerline_render

SRC := \
	core/dataset/trajectory.c \
	core/tokenizer/int_bins.c \
	core/train/lookup_policy.c \
	core/decoder/discrete_action.c \
	core/runtime/runtime.c \
	projects/centerline/centerline.c

RAYLIB_DIR ?= $(firstword $(wildcard raylib-5.5_linux_amd64 /home/claude/pathfinder/raylib-5.5_linux_amd64))
RAYLIB_CFLAGS ?= -I$(RAYLIB_DIR)/include -DPLATFORM_DESKTOP
RAYLIB_LIBS ?= $(RAYLIB_DIR)/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt -lX11

.PHONY: test centerline-render clean

test: $(TEST_BIN)
	./$(TEST_BIN)

centerline-render: $(CENTERLINE_RENDER_BIN)
	./$(CENTERLINE_RENDER_BIN)

$(TEST_BIN): tests/test_centerline_phase0.c $(SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

$(CENTERLINE_RENDER_BIN): apps/centerline_render.c $(SRC)
	mkdir -p build
	test -n "$(RAYLIB_DIR)"
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) $^ $(RAYLIB_LIBS) -o $@

clean:
	rm -rf build
