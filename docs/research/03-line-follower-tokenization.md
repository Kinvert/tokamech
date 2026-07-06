# Line-Follower Tokenization Notes

Research snapshot: 2026-07-06.

## Main takeaway

Use the existing line-follower simulation idea to generate token-sequence data.

The v0 goal should be:

```text
hand-coded sim policy or oracle -> rollout logs -> tokenized dataset -> behavior cloning model -> sim policy replay
```

The robot should come later, after the simulator/tokenizer/dataset/training loop is reliable.

## Observation representation

A useful raw observation frame:

```c
typedef struct {
    uint16_t qti[4];
    uint8_t qti_bin[4];
    int16_t line_pos;
    int16_t line_conf;
    int16_t prev_left_motor;
    int16_t prev_right_motor;
} LineObs;
```

Minimal v0 input:

```text
left_outer
left_inner
right_inner
right_outer
previous_action
```

Prefer calibrated raw reflectance over binary-only sensors. Binary QTI is easy, but raw or quantized raw values preserve edge information that helps sim2real.

## Observation token options

Simplest:

```text
4 sensors x 2 bits each = 8-bit packed sensor token
obs_token range = 0..255
```

More expressive:

```text
4 sensors x 4 bits each = 16-bit packed sensor token
obs_token range = 0..65535
```

More structured:

```text
one token per sensor reading
sensor_0 token range = 0..15
sensor_1 token range = 0..15
sensor_2 token range = 0..15
sensor_3 token range = 0..15
```

Recommended v0:

```text
packed 8-bit observation token
```

Reason:

It is explainable, tiny, C-friendly, and enough to prove the full loop.

## Action representation

Recommended v0 action vocabulary:

```text
0 stop
1 hard_left
2 left
3 slight_left
4 straight
5 slight_right
6 right
7 hard_right
```

Each action maps to a fixed motor command:

```text
stop         -> (0, 0)
hard_left    -> (-turn, +turn)
left         -> (base - d2, base + d2)
slight_left  -> (base - d1, base + d1)
straight     -> (base, base)
slight_right -> (base + d1, base - d1)
right        -> (base + d2, base - d2)
hard_right   -> (+turn, -turn)
```

Alternative later:

```text
left_motor_token in small PWM vocabulary
right_motor_token in small PWM vocabulary
```

Use steering tokens first. They are safer, easier to debug, and easier to review.

## Dataset frame

Human-readable early format:

```json
{
  "episode": 7,
  "t": 123,
  "source": "sim",
  "track_id": "oval_01",
  "qti_raw": [812, 644, 210, 188],
  "obs_token": 57,
  "prev_action": 4,
  "action": 3,
  "reward": 0.81,
  "terminal": false,
  "flags": ["ok"]
}
```

Compact binary format later:

```c
typedef struct {
    uint32_t episode;
    uint32_t t;
    uint16_t qti[4];
    uint16_t obs_token;
    uint8_t prev_action;
    uint8_t action;
    int16_t reward_milli;
    uint8_t done;
    uint8_t flags;
} LineLogFrame;
```

## Token sequence shape

Use typed tokens rather than one untyped global vocabulary.

Example:

```text
BOS
OBS(57)
PREV_ACTION(4)
ACTION(3)
OBS(61)
PREV_ACTION(3)
ACTION(2)
EOS
```

Training view:

```text
input: recent OBS and previous ACTION tokens
target: next ACTION token
```

The simplest training target is only the next action. Predicting the next observation can be added later as an auxiliary task.

## Simulator requirements

The simulator needs:

```text
2D differential-drive kinematics
QTI sensor positions relative to chassis
line/ground reflectance sampling
sensor noise
threshold drift
motor deadband
motor saturation
asymmetric motor gain
fixed control period
```

Recommended step API:

```text
reset(seed, track_id) -> observation
step(action_token) -> observation, reward, done, info
```

## Data generation answer

Yes: the existing PufferLib-style line-follower environment should generate the data.

The clean path is:

```text
1. run simulator with oracle/PD controller
2. at every step, log raw observation and oracle action
3. tokenize observation and action through project tokenizer
4. store fixed episode logs
5. train policy to imitate oracle actions from recent token history
6. run learned policy back inside simulator
```

This gives useful data immediately without requiring the real robot or reinforcement learning.

## Hardware deployment constraints

When deploying later, assume:

```text
no malloc in control loop
no floating point in control loop
fixed-size arrays
integer normalization
small lookup table or tiny int8 neural net
deterministic loop timing
watchdog-safe motor output
non-blocking serial logging
```

Policy loop shape:

```c
while (armed) {
    read_qti(raw);
    obs_token = tokenize_qti(raw, calibration);
    action = policy(obs_token, history);
    action = safety_filter(action, raw);
    set_motors(action_to_pwm[action]);
    log_optional();
    sleep_until_next_tick();
}
```

## Safety layer

The learned policy should suggest actions. It should not bypass safety.

Minimum safety rules:

```text
stop or slow search if all sensors see background for N frames
clamp PWM to safe bounds
rate-limit action changes
require explicit armed state before motors move
stop on watchdog timeout
allow serial kill switch
run startup calibration before enabling policy
```

## Minimal milestone path

```text
1. Sim oracle follows a simple simulated track.
2. Oracle emits discrete steering tokens.
3. Simulator writes tokenized episode logs.
4. Tiny model learns next action from token history.
5. Learned model runs closed-loop in sim.
6. C tokenizer replays saved logs exactly.
7. Real robot logs QTI with motors disabled.
8. Calibration aligns real sensor distribution to sim tokens.
9. Tethered low-speed robot test with kill switch.
10. Untethered hello-world run.
```
