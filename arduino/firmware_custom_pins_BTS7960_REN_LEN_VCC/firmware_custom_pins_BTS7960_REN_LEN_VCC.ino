#include <Arduino.h>
#include <math.h>

/*
X(BLOCK) = BLOCK * 638 + 420
Y(BLOCK) = BLOCK * 638
Z(DEGREE) = (DEGREE / 90) * 400
  =====================================================================
  Custom Board Motion Firmware (UNO R3)
  ---------------------------------------------------------------------
  在原版多轴 STEP/DIR 运动固件的基础上，按"控制板引脚定义文档"做了如下改造：

    - 保留 X / Y / Z 三轴的梯形加减速运动控制（删除原 W 轴）
        Axis X = MOTORX 外部驱动器       (PUX-/DRX-，光耦共阳)
        Axis Y = MOTORY 外部驱动器       (PUY-/DRY-，光耦共阳)
        Axis Z = 板载 TMC2209 静音驱动器  (Step/Dir，逻辑电平直驱)

    - 集成 4 路光电限位（NPN 开漏，接 INPUT_PULLUP）
        X 轴：LSX1 = 负向限位，LSX2 = 正向限位
        Y 轴：LSY1 = 负向限位，LSY2 = 正向限位
        Z 轴：无限位（板载电机内部机构）

    - 新增 H 桥直流电机控制（BTS7960 / RPWM-LPWM 双 PWM 半桥）
        与步进运动相互独立，不参与梯形加减速。
        本版按指定引脚使用 RPWM=D3、LPWM=D2。由于 UNO 的 D2 不是硬件 PWM，
        固件使用 Timer2 生成软件 PWM；步进调度改用 Timer1。
        BTS7960 的 R_EN / L_EN 已硬接 VCC，软件不再控制 EN 引脚。

    - 去除原固件的 PIN_EN_ALL 使能引脚
        板载 TMC2209 的 EN 已硬连到 GND，软件无法控制；外部驱动器的
        ENA 也未引出到本板。EN 改为软标志，仅用于状态查询。

  ---------------------------------------------------------------------
  串口协议 (115200 baud, '\n' 结尾)
  ---------------------------------------------------------------------
    HELP                       帮助
    PING                       -> PONG
    STATUS?                    返回三轴位置 + DC 状态 + BUSY
    LIMITS?                    返回 4 路限位实时状态
    INIT                       X/Y 回负限位，X 执行 BM X 2
    AUTO_PUSH                  自动推送流程，结束后执行 MOVE X -10000 复位

    MOVE X|Y|Z <steps> F <rate> [A <acc> | AU <up> AD <down>]
        steps  : 整数步数（带符号）
        F      : 最大步频 (步/秒)
        A      : 对称加减速度（步/秒²）
        AU/AD  : 非对称加速度/减速度
    STOP                       所有轴减速停止
    STOP!                      所有轴急停（含 DC）

    DC F <0-255>               H 桥直流电机正转，PWM 占空比
    DC R <0-255>               反转
    DC 0  /  DC STOP           停止

    EN 0|1                     软使能标志（仅状态用，板载 TMC2209 永远使能）

  响应：OK / ERR <msg> / DONE <axis> [LIMIT] / LIMIT <axis>±
  =====================================================================
*/

// ============================================================================
// 引脚映射  —  按用户指定 Arduino 引脚定义
// ============================================================================
// 注意：Arduino UNO 上 D2 不是硬件 PWM。由于本接线要求 PIN_LPWM = D2，
// 本固件没有对 BTS7960 使用 analogWrite()，而是用 Timer2 生成软件 PWM。
// 步进运动调度改由 Timer1 完成；因此 D9/D10 可作为普通 STEP / 限位输入使用。
// 另外，D13 被用作 Z_DIR，会与 UNO 板载 LED 共用，因此固件禁用心跳灯输出，
// 避免心跳灯翻转影响 Z 轴方向。

// --- 板载 TMC2209（Z 轴）---
static const uint8_t PIN_Z_DIR  = 13;   // ARDUINO-1  -> U1-Dir
static const uint8_t PIN_Z_STEP = 12;   // ARDUINO-2  -> U1-Step

// --- 外部 X 驱动器（MOTORX，光耦共阳）---
static const uint8_t PIN_X_STEP = 9;    // ARDUINO-5  -> PUX-
static const uint8_t PIN_X_DIR  = 8;    // ARDUINO-6  -> DRX-

// --- 外部 Y 驱动器（MOTORY，光耦共阳）---
static const uint8_t PIN_Y_STEP = 5;    // ARDUINO-9  -> PUY-
static const uint8_t PIN_Y_DIR  = 4;    // ARDUINO-10 -> DRY-

// --- H 桥直流电机（H2）---
// BTS7960 的 R_EN / L_EN 已接 VCC，因此 Arduino 只输出 RPWM / LPWM。
// PIN_LPWM = D2 在 UNO 上不是硬件 PWM，本固件用 Timer2 软件 PWM 驱动。
static const uint8_t PIN_RPWM = 3;      // ARDUINO-11 -> BTS7960 RPWM，Timer2 软件 PWM
static const uint8_t PIN_LPWM = 2;      // ARDUINO-12 -> BTS7960 LPWM，Timer2 软件 PWM

// --- 4 路光电限位（NPN 开漏，配合 INPUT_PULLUP）---
static const uint8_t PIN_LSX1 = 11;     // ARDUINO-3  -> LSX1  X 负向
static const uint8_t PIN_LSX2 = 10;     // ARDUINO-4  -> LSX2  X 正向
static const uint8_t PIN_LSY1 = 7;      // ARDUINO-7  -> LSY1  Y 负向
static const uint8_t PIN_LSY2 = 6;      // ARDUINO-8  -> LSY2  Y 正向

static const uint8_t LIMIT_ACTIVE_LEVEL = LOW;   // 触发时的电平
#define NO_PIN 255

// ----------------------------------------------------------------------------
// X/Y 外部 2D45A 驱动方式：来自 stepper_2D45A_with_limits 示例
// ----------------------------------------------------------------------------
// 2D45A 按共阳极接法使用：Arduino 5V -> PU+/DR+，Arduino 引脚 -> PU-/DR-。
// 因此 X/Y 的 STEP 空闲态为 HIGH，发步进脉冲时拉 LOW 再回 HIGH；
// 方向也沿用示例中的定义：正向 = LOW，反向 = HIGH。
// Z 轴仍保持板载 TMC2209 的普通逻辑电平：STEP 空闲 LOW，脉冲 HIGH。
static const uint16_t STEP_PULSE_WIDTH_US = 4;    // ISR 中的有效脉冲宽度

// ============================================================================
// 运动参数 —— 与原固件一致
// ============================================================================
static const uint32_t SERIAL_BAUD       = 115200;
static const uint16_t MAX_STEP_RATE     = 3000;
static const uint16_t MIN_STEP_RATE     = 1;
static const uint16_t TIMER1_PRESCALER  = 8;
static const uint16_t SCHED_TICK_HZ     = 12000;   // 必须 > MAX_STEP_RATE；Timer1 负责步进调度

static const uint8_t  RATE_FP_SHIFT = 16;
static const uint32_t RATE_FP_ONE   = (1UL << RATE_FP_SHIFT);

static const float UPDATE_DT_MIN = 0.002f;
static const float STOP_EPS_V    = 1.0f;

static const long     INIT_HOME_STEPS  = 1000000L;
static const uint16_t INIT_HOME_RATE   = 800;
static const float    INIT_HOME_ACCEL  = 600.0f;
static const long     INIT_X_OFFSET    = 420L;

static const long     AUTO_PUSH_BM_X_STEPS = 638L;
static const long     AUTO_PUSH_RESET_X_STEPS = -10000L;
static const uint8_t  AUTO_PUSH_MAX_TRIES   = 3;
static const uint32_t AUTO_PUSH_DC_2S_MS    = 2500UL;
static const uint32_t AUTO_PUSH_DC_15S_MS   = 5000UL;
static const uint32_t AUTO_PUSH_WAIT_MS     = 2000UL;
static const uint32_t A0_TEST_INTERVAL_MS   = 100UL;

// ============================================================================
// 轴结构 —— 在原版基础上加了 limit_neg/pos 与 limit_hit
// ============================================================================
struct Axis {
  char    name;
  uint8_t pin_step;
  uint8_t pin_dir;
  uint8_t pin_limit_neg;   // 负向限位引脚（NO_PIN 表示无）
  uint8_t pin_limit_pos;   // 正向限位引脚

  // STEP/DIR 电平策略。X/Y 使用 2D45A 共阳极低脉冲；Z 使用 TMC2209 高脉冲。
  uint8_t step_idle_level;
  uint8_t step_active_level;
  uint8_t dir_positive_level;

  volatile long  pos_steps;
  volatile long  remaining_steps;
  volatile bool  dir_positive;
  volatile bool  active;
  volatile bool  stop_requested;
  volatile bool  limit_hit;        // 本次 MOVE 是否被限位中止

  volatile float target_v;
  volatile float accel_up;
  volatile float accel_down;
  volatile float current_v;
  volatile float planned_v;

  volatile uint32_t rate_fp;
  volatile uint32_t phase_fp;
};

static Axis AXIS_X = {'X', PIN_X_STEP, PIN_X_DIR, PIN_LSX1, PIN_LSX2,
                      HIGH, LOW, LOW,
                      0, 0, true, false, false, false,
                      800.0f, 600.0f, 600.0f, 0.0f, 0.0f, 0, 0};
static Axis AXIS_Y = {'Y', PIN_Y_STEP, PIN_Y_DIR, PIN_LSY1, PIN_LSY2,
                      HIGH, LOW, LOW,
                      0, 0, true, false, false, false,
                      800.0f, 600.0f, 600.0f, 0.0f, 0.0f, 0, 0};
static Axis AXIS_Z = {'Z', PIN_Z_STEP, PIN_Z_DIR, NO_PIN,   NO_PIN,
                      LOW, HIGH, HIGH,
                      0, 0, true, false, false, false,
                      800.0f, 600.0f, 600.0f, 0.0f, 0.0f, 0, 0};

static Axis* ACTIVE_AXES[] = { &AXIS_X, &AXIS_Y, &AXIS_Z };
static const uint8_t NUM_AXES = sizeof(ACTIVE_AXES) / sizeof(ACTIVE_AXES[0]);

// 全局状态
static volatile bool g_enabled = true;     // 软使能标志
static int16_t       g_dc_speed = 0;       // -255..+255
static volatile uint8_t g_dc_pwm_duty = 0;   // 0..255，Timer2 软件 PWM 占空比
static volatile int8_t  g_dc_pwm_dir  = 0;   // +1=RPWM，-1=LPWM，0=停止

enum InitState {
  INIT_IDLE,
  INIT_HOMING_XY,
  INIT_X_OFFSET_MOVE
};

static InitState g_init_state = INIT_IDLE;

enum AutoPushState {
  AUTO_PUSH_IDLE,
  AUTO_PUSH_HOME_X,
  AUTO_PUSH_X_OFFSET,
  AUTO_PUSH_DC_80,
  AUTO_PUSH_DC_NEG_80,
  AUTO_PUSH_BM_X,
  AUTO_PUSH_DC_180,
  AUTO_PUSH_WAIT,
  AUTO_PUSH_DC_NEG_180,
  AUTO_PUSH_FINAL_DC_NEG_80,
  AUTO_PUSH_RESET_X
};

static AutoPushState g_auto_push_state = AUTO_PUSH_IDLE;
static uint32_t g_auto_push_stage_ms = 0;
static uint8_t g_auto_push_tries = 0;
static bool g_auto_push_done_max_after_reset = false;
static volatile bool g_auto_push_ignore_x_pos_limit = false;
static bool g_bm_z_done_pending = false;

static bool g_a0_test_enabled = false;
static uint32_t g_a0_test_last_ms = 0;

// 串口缓冲
static char    linebuf[96];
static uint8_t linepos = 0;

// ============================================================================
// 辅助函数
// ============================================================================
static void reply_ok(){ Serial.println(F("OK")); }
static void reply_err(const __FlashStringHelper* msg){
  Serial.print(F("ERR "));
  Serial.println(msg);
}

static bool any_axis_busy(){
  for(uint8_t i=0; i<NUM_AXES; i++){
    if(ACTIVE_AXES[i]->active) return true;
  }
  return false;
}

static uint32_t speed_to_fp(float steps_per_sec){
  if(steps_per_sec <= 0.0f) return 0;
  if(steps_per_sec < (float)MIN_STEP_RATE) steps_per_sec = (float)MIN_STEP_RATE;
  if(steps_per_sec > (float)MAX_STEP_RATE) steps_per_sec = (float)MAX_STEP_RATE;
  float steps_per_tick = steps_per_sec / (float)SCHED_TICK_HZ;
  uint32_t fp = (uint32_t)(steps_per_tick * (float)RATE_FP_ONE + 0.5f);
  if(fp == 0) fp = 1;
  return fp;
}

static inline bool limit_active(uint8_t pin){
  if(pin == NO_PIN) return false;
  return digitalRead(pin) == LIMIT_ACTIVE_LEVEL;
}

static inline uint8_t invert_level(uint8_t level){
  return (level == HIGH) ? LOW : HIGH;
}

static inline void step_idle(Axis* ax){
  digitalWrite(ax->pin_step, ax->step_idle_level);
}

static inline void write_dir(Axis* ax, bool dir_positive){
  digitalWrite(ax->pin_dir, dir_positive ? ax->dir_positive_level : invert_level(ax->dir_positive_level));
}

static inline void emit_step_pulse(Axis* ax){
  digitalWrite(ax->pin_step, ax->step_active_level);
  delayMicroseconds(STEP_PULSE_WIDTH_US);
  digitalWrite(ax->pin_step, ax->step_idle_level);
}

// ============================================================================
// 定时器初始化
// ============================================================================
// Timer1：步进运动调度。当前引脚表中 D9/D10 不再用于 DC PWM，
// 因此 Timer1 可以安全用于步进 ISR；D9 仍可作为普通 STEP 输出。
static void timer1_start_step_scheduler(){
  uint32_t tick = F_CPU / TIMER1_PRESCALER;
  uint32_t ocr  = tick / (uint32_t)SCHED_TICK_HZ;
  if(ocr == 0)      ocr = 1;
  if(ocr > 65536UL) ocr = 65536UL;   // Timer1 是 16 位定时器
  ocr--;

  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A  = (uint16_t)ocr;
  TCCR1B |= (1 << WGM12);            // CTC 模式
  TCCR1B |= (1 << CS11);             // 分频 = 8
  TIMSK1 |= (1 << OCIE1A);
  interrupts();
}

// Timer2：BTS7960 RPWM/LPWM 软件 PWM。
// UNO 的 D2 不是硬件 PWM，因此不能对 PIN_LPWM 直接 analogWrite()。
// 这里使用 Timer2 Fast PWM 的 overflow/compare 中断生成约 976 Hz 的 8-bit PWM。
static void timer2_start_dc_pwm(){
  noInterrupts();
  TCCR2A = 0;
  TCCR2B = 0;
  TCNT2  = 0;
  OCR2A  = 0;
  TCCR2A |= (1 << WGM20) | (1 << WGM21); // Fast PWM, TOP=0xFF
  TCCR2B |= (1 << CS22);                 // 分频 = 64，PWM 约 976 Hz
  TIMSK2 |= (1 << TOIE2);                // 周期开始时拉高有效 PWM 引脚
  TIMSK2 &= ~(1 << OCIE2A);              // duty=0/255 时不需要比较中断
  interrupts();
}

static inline void dc_pwm_update_compare(uint8_t duty){
  if(duty == 0 || duty >= 255){
    TIMSK2 &= ~(1 << OCIE2A);
  } else {
    OCR2A = duty;
    TIFR2 |= (1 << OCF2A);               // 清除可能残留的比较标志
    TIMSK2 |= (1 << OCIE2A);
  }
}

// ============================================================================
// H 桥直流电机
// ============================================================================
static void dc_set_speed(int16_t speed){
  // R_EN / L_EN 已硬接 VCC，软件只控制 RPWM / LPWM。
  // 由于 PIN_LPWM=D2 不是 UNO 硬件 PWM，本函数不使用 analogWrite()。
  if(speed > 255)  speed = 255;
  if(speed < -255) speed = -255;

  uint8_t duty = (speed >= 0) ? (uint8_t)speed : (uint8_t)(-speed);
  int8_t dir = 0;
  if(speed > 0)      dir = +1;
  else if(speed < 0) dir = -1;

  noInterrupts();
  g_dc_pwm_dir  = 0;       // 先停 PWM，避免切向瞬间两路同时有效
  g_dc_pwm_duty = 0;
  dc_pwm_update_compare(0);
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);

  g_dc_pwm_duty = duty;
  g_dc_pwm_dir  = dir;
  dc_pwm_update_compare(duty);
  interrupts();

  g_dc_speed = speed;
}

static void dc_stop(){
  noInterrupts();
  g_dc_pwm_dir  = 0;
  g_dc_pwm_duty = 0;
  dc_pwm_update_compare(0);
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);
  interrupts();
  g_dc_speed = 0;
}

// ============================================================================
// 查找轴
// ============================================================================
static Axis* find_axis(char c){
  c = toupper(c);
  for(uint8_t i=0; i<NUM_AXES; i++){
    if(ACTIVE_AXES[i]->name == c) return ACTIVE_AXES[i];
  }
  return nullptr;
}

// ============================================================================
// 启动 / 停止
// ============================================================================
static bool start_move(Axis* ax, long steps, uint16_t rate, float aup, float adown){
  if(!ax) return false;
  if(steps == 0) return true;

  bool dir = (steps >= 0);

  // ---- 起步前限位检查：如果已经压在该方向限位上，拒绝起步 ----
  uint8_t blocking = dir ? ax->pin_limit_pos : ax->pin_limit_neg;
  bool ignore_limit = (ax == &AXIS_X && dir && g_auto_push_ignore_x_pos_limit);
  if(!ignore_limit && limit_active(blocking)){
    Serial.print(F("ERR limit_already_active "));
    Serial.print(ax->name);
    Serial.println(dir ? F("+") : F("-"));
    return false;
  }

  long abs_steps = (steps >= 0) ? steps : -steps;

  pinMode(ax->pin_dir, OUTPUT);
  pinMode(ax->pin_step, OUTPUT);
  step_idle(ax);
  write_dir(ax, dir);

  noInterrupts();
  ax->remaining_steps = abs_steps;
  ax->dir_positive    = dir;
  ax->target_v        = (float)rate;
  ax->accel_up        = aup;
  ax->accel_down      = adown;
  ax->current_v       = 0.0f;
  ax->planned_v       = 0.0f;
  ax->stop_requested  = false;
  ax->limit_hit       = false;
  ax->rate_fp         = 0;
  ax->phase_fp        = 0;
  ax->active          = (abs_steps > 0);
  interrupts();
  return true;
}

static void request_stop_all(bool send_ok){
  g_init_state = INIT_IDLE;
  if(g_auto_push_state != AUTO_PUSH_IDLE){
    dc_stop();
  }
  g_auto_push_state = AUTO_PUSH_IDLE;
  g_bm_z_done_pending = false;
  // g_auto_push_ignore_x_pos_limit = false;

  noInterrupts();
  for(uint8_t i=0; i<NUM_AXES; i++){
    if(ACTIVE_AXES[i]->active){
      ACTIVE_AXES[i]->stop_requested = true;
    }
  }
  interrupts();
  if(send_ok) reply_ok();
}

static void emergency_stop_all(bool send_ok){
  g_init_state = INIT_IDLE;
  g_auto_push_state = AUTO_PUSH_IDLE;
  g_bm_z_done_pending = false;
  // g_auto_push_ignore_x_pos_limit = false;

  noInterrupts();
  for(uint8_t i=0; i<NUM_AXES; i++){
    ACTIVE_AXES[i]->remaining_steps = 0;
    ACTIVE_AXES[i]->active          = false;
    ACTIVE_AXES[i]->stop_requested  = false;
    ACTIVE_AXES[i]->current_v       = 0.0f;
    ACTIVE_AXES[i]->planned_v       = 0.0f;
    ACTIVE_AXES[i]->phase_fp        = 0;
    ACTIVE_AXES[i]->rate_fp         = 0;
    step_idle(ACTIVE_AXES[i]);
  }
  interrupts();
  dc_stop();
  if(send_ok) reply_ok();
}

static bool axis_active(Axis* ax){
  noInterrupts();
  bool active = ax->active;
  interrupts();
  return active;
}

static void set_axis_position(Axis* ax, long pos){
  noInterrupts();
  ax->pos_steps = pos;
  interrupts();
}

static bool axis_negative_limit_active(Axis* ax){
  return limit_active(ax->pin_limit_neg);
}

static bool initialize_axis_home(Axis* ax){
  if(axis_negative_limit_active(ax)){
    set_axis_position(ax, 0);
    return true;
  }
  return start_move(ax, -INIT_HOME_STEPS, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL);
}

static void initialize(){
  if(g_init_state != INIT_IDLE){
    reply_err(F("init_busy"));
    return;
  }
  if(g_auto_push_state != AUTO_PUSH_IDLE){
    reply_err(F("auto_push_busy"));
    return;
  }
  if(any_axis_busy()){
    reply_err(F("busy"));
    return;
  }

  if(!initialize_axis_home(&AXIS_X)){
    reply_err(F("init_x_start_failed"));
    return;
  }
  if(!initialize_axis_home(&AXIS_Y)){
    emergency_stop_all(false);
    reply_err(F("init_y_start_failed"));
    return;
  }

  g_init_state = INIT_HOMING_XY;
  reply_ok();
}

static void update_initialize(){
  if(g_init_state == INIT_IDLE) return;

  if(g_init_state == INIT_HOMING_XY){
    if(axis_active(&AXIS_X) || axis_active(&AXIS_Y)) return;

    if(!axis_negative_limit_active(&AXIS_X)){
      g_init_state = INIT_IDLE;
      reply_err(F("init_x_limit_not_reached"));
      return;
    }
    if(!axis_negative_limit_active(&AXIS_Y)){
      g_init_state = INIT_IDLE;
      reply_err(F("init_y_limit_not_reached"));
      return;
    }

    set_axis_position(&AXIS_X, 0);
    set_axis_position(&AXIS_Y, 0);

    if(!start_move(&AXIS_X, INIT_X_OFFSET + 638, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL)){
      g_init_state = INIT_IDLE;
      reply_err(F("init_x_offset_failed"));
      return;
    }

    g_init_state = INIT_X_OFFSET_MOVE;
    return;
  }

  if(g_init_state == INIT_X_OFFSET_MOVE){
    if(axis_active(&AXIS_X)) return;

    if(limit_active(AXIS_X.pin_limit_pos)){
      g_init_state = INIT_IDLE;
      reply_err(F("init_x_offset_limit"));
      return;
    }

    g_init_state = INIT_IDLE;
    Serial.println(F("INIT DONE"));
  }
}

static bool elapsed_ms(uint32_t started_ms, uint32_t duration_ms){
  return (uint32_t)(millis() - started_ms) >= duration_ms;
}

static bool auto_push_start_x_home(){
  if(axis_negative_limit_active(&AXIS_X)){
    set_axis_position(&AXIS_X, 0);
    return true;
  }
  return start_move(&AXIS_X, -INIT_HOME_STEPS, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL);
}

static void auto_push_fail(const __FlashStringHelper* msg){
  dc_stop();
  g_auto_push_state = AUTO_PUSH_IDLE;
  g_auto_push_done_max_after_reset = false;
  // g_auto_push_ignore_x_pos_limit = false;
  reply_err(msg);
}

static void auto_push_finish_after_reset(){
  g_auto_push_state = AUTO_PUSH_IDLE;
  // g_auto_push_ignore_x_pos_limit = false;
  if(g_auto_push_done_max_after_reset){
    g_auto_push_done_max_after_reset = false;
    Serial.println(F("AUTO_PUSH FAILED MAX"));
  } else {
    Serial.println(F("AUTO_PUSH SUCCESS"));
  }
}

static void auto_push_start_reset(bool done_max){
  dc_stop();
  g_auto_push_done_max_after_reset = done_max;

  if(axis_negative_limit_active(&AXIS_X)){
    set_axis_position(&AXIS_X, 0);
    auto_push_finish_after_reset();
    return;
  }

  if(!start_move(&AXIS_X, AUTO_PUSH_RESET_X_STEPS, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL)){
    auto_push_fail(F("auto_push_reset_failed"));
    return;
  }

  g_auto_push_state = AUTO_PUSH_RESET_X;
}

static void auto_push_done_max(){
  auto_push_start_reset(true);
}

static void auto_push_start_probe(){
  if(g_auto_push_tries >= AUTO_PUSH_MAX_TRIES){
    auto_push_done_max();
    return;
  }
  g_auto_push_tries++;
  dc_set_speed(80);
  g_auto_push_stage_ms = millis();
  g_auto_push_state = AUTO_PUSH_DC_80;
}

static void auto_push(){
  if(g_auto_push_state != AUTO_PUSH_IDLE){
    reply_err(F("auto_push_busy"));
    return;
  }
  if(g_init_state != INIT_IDLE){
    reply_err(F("init_busy"));
    return;
  }
  if(any_axis_busy()){
    reply_err(F("busy"));
    return;
  }

  if(!auto_push_start_x_home()){
    reply_err(F("auto_push_home_start_failed"));
    return;
  }

  g_auto_push_tries = 0;
  // g_auto_push_ignore_x_pos_limit = true;
  g_auto_push_state = AUTO_PUSH_HOME_X;
  reply_ok();
}

static void update_auto_push(){
  if(g_auto_push_state == AUTO_PUSH_IDLE) return;

  if(g_auto_push_state == AUTO_PUSH_HOME_X){
    if(axis_active(&AXIS_X)) return;

    if(!axis_negative_limit_active(&AXIS_X)){
      auto_push_fail(F("auto_push_x_limit_not_reached"));
      return;
    }

    set_axis_position(&AXIS_X, 0);
    if(!start_move(&AXIS_X, INIT_X_OFFSET, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL)){
      auto_push_fail(F("auto_push_x_offset_failed"));
      return;
    }

    g_auto_push_state = AUTO_PUSH_X_OFFSET;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_X_OFFSET){
    if(axis_active(&AXIS_X)) return;

    auto_push_start_probe();
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_DC_80){
    if(analogRead(A0) == 0){
      delay(10);
      if (analogRead(A0) == 0) {
        dc_stop();
        dc_set_speed(-180);
        g_auto_push_stage_ms = millis();
        g_auto_push_state = AUTO_PUSH_DC_NEG_80;
        return;
      }
    }

    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_DC_2S_MS)) return;

    dc_set_speed(180);
    g_auto_push_stage_ms = millis();
    g_auto_push_state = AUTO_PUSH_DC_180;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_DC_NEG_80){
    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_DC_2S_MS)) return;

    dc_stop();
    if(!start_move(&AXIS_X, AUTO_PUSH_BM_X_STEPS, INIT_HOME_RATE, INIT_HOME_ACCEL, INIT_HOME_ACCEL)){
      auto_push_fail(F("auto_push_bm_failed"));
      return;
    }

    g_auto_push_state = AUTO_PUSH_BM_X;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_BM_X){
    if(axis_active(&AXIS_X)) return;

    auto_push_start_probe();
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_DC_180){
    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_DC_15S_MS)) return;

    dc_stop();
    g_auto_push_stage_ms = millis();
    g_auto_push_state = AUTO_PUSH_WAIT;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_WAIT){
    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_WAIT_MS)) return;

    dc_set_speed(-180);
    g_auto_push_stage_ms = millis();
    g_auto_push_state = AUTO_PUSH_DC_NEG_180;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_DC_NEG_180){
    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_DC_15S_MS)) return;

    dc_set_speed(-180);
    g_auto_push_stage_ms = millis();
    g_auto_push_state = AUTO_PUSH_FINAL_DC_NEG_80;
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_FINAL_DC_NEG_80){
    if(!elapsed_ms(g_auto_push_stage_ms, AUTO_PUSH_DC_2S_MS)) return;

    auto_push_start_reset(false);
    return;
  }

  if(g_auto_push_state == AUTO_PUSH_RESET_X){
    if(axis_active(&AXIS_X)) return;

    if(axis_negative_limit_active(&AXIS_X)){
      set_axis_position(&AXIS_X, 0);
    }

    auto_push_finish_after_reset();
  }
}

// ============================================================================
// ISR：12kHz 调度，定点累加发脉冲
// X/Y 已替换为 stepper_2D45A 的 2D45A 共阳极低脉冲方式；Z 保持 TMC2209 高脉冲。
// ============================================================================
ISR(TIMER1_COMPA_vect){
  for(uint8_t i=0; i<NUM_AXES; i++){
    Axis* ax = ACTIVE_AXES[i];
    if(!ax->active) continue;
    if(ax->rate_fp == 0) continue;
    if(ax->remaining_steps <= 0){
      ax->active = false;
      continue;
    }

    ax->phase_fp += ax->rate_fp;
    if(ax->phase_fp < RATE_FP_ONE) continue;
    ax->phase_fp -= RATE_FP_ONE;

    emit_step_pulse(ax);

    ax->remaining_steps--;
    if(ax->dir_positive) ax->pos_steps++;
    else                 ax->pos_steps--;

    if(ax->remaining_steps <= 0){
      ax->active = false;
    }
  }
}


// Timer2 overflow：开始一个新的 DC PWM 周期
ISR(TIMER2_OVF_vect){
  uint8_t duty = g_dc_pwm_duty;
  int8_t dir   = g_dc_pwm_dir;

  if(dir == 0 || duty == 0){
    digitalWrite(PIN_RPWM, LOW);
    digitalWrite(PIN_LPWM, LOW);
    return;
  }

  if(dir > 0){
    digitalWrite(PIN_LPWM, LOW);
    digitalWrite(PIN_RPWM, HIGH);
  } else {
    digitalWrite(PIN_RPWM, LOW);
    digitalWrite(PIN_LPWM, HIGH);
  }

  // duty=255 时保持整周期高电平，不启用 compare 拉低。
}

// Timer2 compare：达到 duty 后拉低当前有效 PWM 引脚
ISR(TIMER2_COMPA_vect){
  int8_t dir = g_dc_pwm_dir;
  if(dir > 0){
    digitalWrite(PIN_RPWM, LOW);
  } else if(dir < 0){
    digitalWrite(PIN_LPWM, LOW);
  } else {
    digitalWrite(PIN_RPWM, LOW);
    digitalWrite(PIN_LPWM, LOW);
  }
}

// ============================================================================
// 梯形规划 + 限位检测  —— 核心改动点
// ============================================================================
static void update_motion(){
  static uint32_t last_us = 0;
  uint32_t now = micros();
  if(last_us == 0) last_us = now;
  float dt = (now - last_us) / 1000000.0f;
  if(dt < UPDATE_DT_MIN) return;
  last_us = now;

  for(uint8_t i=0; i<NUM_AXES; i++){
    Axis* ax = ACTIVE_AXES[i];
    if(!ax->active) continue;

    long  rem_steps;
    float target_v, accel_up, accel_down, current_v;
    bool  stop_requested, dir_positive;

    noInterrupts();
    rem_steps      = ax->remaining_steps;
    target_v       = ax->target_v;
    accel_up       = ax->accel_up;
    accel_down     = ax->accel_down;
    current_v      = ax->current_v;
    stop_requested = ax->stop_requested;
    dir_positive   = ax->dir_positive;
    interrupts();

    // -------- 限位检查（只检查当前行进方向那一侧）--------
    uint8_t check_pin = dir_positive ? ax->pin_limit_pos : ax->pin_limit_neg;
    bool ignore_limit = (ax == &AXIS_X && dir_positive && g_auto_push_ignore_x_pos_limit);
    if(!ignore_limit && limit_active(check_pin)){
      noInterrupts();
      ax->remaining_steps = 0;
      ax->active          = false;
      ax->stop_requested  = false;
      ax->current_v       = 0.0f;
      ax->planned_v       = 0.0f;
      ax->rate_fp         = 0;
      ax->limit_hit       = true;
      interrupts();

      Serial.print(F("LIMIT "));
      Serial.print(ax->name);
      Serial.println(dir_positive ? F("+") : F("-"));
      continue;
    }

    if(rem_steps <= 0){
      noInterrupts();
      ax->current_v      = 0.0f;
      ax->planned_v      = 0.0f;
      ax->rate_fp        = 0;
      ax->active         = false;
      ax->stop_requested = false;
      interrupts();
      continue;
    }

    // -------- 梯形规划（与原固件一致）--------
    float planned_v;
    if(stop_requested){
      planned_v = 0.0f;
    } else {
      float decel_dist = (current_v * current_v) / (2.0f * accel_down);
      if((float)rem_steps <= decel_dist){
        planned_v = sqrt(2.0f * accel_down * (float)rem_steps);
      } else {
        planned_v = target_v;
      }
    }
    if(planned_v > target_v) planned_v = target_v;
    if(planned_v < 0.0f)     planned_v = 0.0f;

    if(current_v < planned_v){
      current_v += accel_up * dt;
      if(current_v > planned_v) current_v = planned_v;
    } else if(current_v > planned_v){
      current_v -= accel_down * dt;
      if(current_v < planned_v) current_v = planned_v;
    }

    if(current_v < STOP_EPS_V && (stop_requested || rem_steps <= 1)){
      current_v = 0.0f;
    }

    noInterrupts();
    ax->current_v = current_v;
    ax->planned_v = planned_v;
    ax->rate_fp   = speed_to_fp(current_v);
    if(current_v <= 0.0f && (stop_requested || rem_steps <= 0)){
      ax->active         = false;
      ax->stop_requested = false;
      ax->rate_fp        = 0;
    }
    interrupts();
  }
}

// ============================================================================
// 命令处理
// ============================================================================
static void print_help(){
  Serial.println(F("=== Custom Board Motion Firmware HELP ==="));
  Serial.println(F("Steppers (3 axes, trapezoidal):"));
  Serial.println(F("  INIT  (homes X/Y, then runs BM X 2)"));
  Serial.println(F("  AUTO_PUSH  (ends with MOVE X -10000 reset)"));
  Serial.println(F("  MOVE X|Y|Z <steps> F <rate> [A <acc> | AU <up> AD <down>]"));
  Serial.println(F("  STOP   STOP!"));
  Serial.println(F("DC motor (BTS7960):"));
  Serial.println(F("  DC F <0-255>   forward"));
  Serial.println(F("  DC R <0-255>   reverse"));
  Serial.println(F("  DC 0           stop"));
  Serial.println(F("Queries:"));
  Serial.println(F("  STATUS?   LIMITS?   A0TEST 0|1   HELP   PING"));
  Serial.println(F("Mapping: X=MOTORX  Y=MOTORY  Z=onboard TMC2209"));
  Serial.println(F("Limits:  X<->LSX1/LSX2   Y<->LSY1/LSY2   Z=none"));
}

static bool starts_with(const char* s, const char* prefix){
  while(*prefix){
    if(toupper(*s) != toupper(*prefix)) return false;
    s++; prefix++;
  }
  return true;
}

static bool extract_axis_from_token(const char* token, char& axis_out){
  for(uint8_t i=0; token[i]; i++){
    char c = toupper(token[i]);
    if(c=='X' || c=='Y' || c=='Z'){
      axis_out = c;
      return true;
    }
  }
  return false;
}

static void cmd_status(){
  Serial.print(F("POS X="));   Serial.print((long)AXIS_X.pos_steps);
  Serial.print(F(" Y="));      Serial.print((long)AXIS_Y.pos_steps);
  Serial.print(F(" Z="));      Serial.print((long)AXIS_Z.pos_steps);
  Serial.print(F(" DC="));     Serial.print(g_dc_speed);
  Serial.print(F(" EN="));     Serial.print(g_enabled ? 1 : 0);
  Serial.print(F(" BUSY="));   Serial.println(any_axis_busy() ? 1 : 0);
}

static void cmd_limits(){
  Serial.print(F("A0=")); Serial.print(analogRead(A0));
  Serial.print(F(" LSX1="));  Serial.print(limit_active(PIN_LSX1) ? 1 : 0);
  Serial.print(F(" LSX2=")); Serial.print(limit_active(PIN_LSX2) ? 1 : 0);
  Serial.print(F(" LSY1=")); Serial.print(limit_active(PIN_LSY1) ? 1 : 0);
  Serial.print(F(" LSY2=")); Serial.println(limit_active(PIN_LSY2) ? 1 : 0);
}

static void update_a0_test(){
  if(!g_a0_test_enabled) return;

  uint32_t now = millis();
  if((uint32_t)(now - g_a0_test_last_ms) < A0_TEST_INTERVAL_MS) return;
  g_a0_test_last_ms = now;

  Serial.print(F("A0="));
  Serial.println(analogRead(A0));
}

static void handle_line(char* line){
  // 去尾部空白
  size_t n = strlen(line);
  while(n > 0 && (line[n-1]=='\r' || line[n-1]=='\n' || line[n-1]==' ' || line[n-1]=='\t')){
    line[--n] = 0;
  }
  if(line[0] == 0) return;

  // 切分 token
  char* tokens[16];
  uint8_t nt = 0;
  char* p = line;
  while(*p && nt < 16){
    while(*p==' ' || *p=='\t') p++;
    if(!*p) break;
    tokens[nt++] = p;
    while(*p && *p!=' ' && *p!='\t') p++;
    if(*p){ *p = 0; p++; }
  }
  if(nt == 0) return;

  if(starts_with(tokens[0], "HELP"))  { print_help();           return; }
  if(starts_with(tokens[0], "PING"))  { Serial.println(F("PONG")); return; }
  if(starts_with(tokens[0], "STOP!")) { emergency_stop_all(true); return; }
  if(starts_with(tokens[0], "STOP"))  { request_stop_all(true);   return; }
  if(starts_with(tokens[0], "STATUS")){ cmd_status();             return; }
  if(starts_with(tokens[0], "LIMITS")){ cmd_limits();             return; }
  if(starts_with(tokens[0], "A0TEST")){
    if(nt < 2){ reply_err(F("A0TEST needs 0/1")); return; }
    g_a0_test_enabled = (atoi(tokens[1]) != 0);
    g_a0_test_last_ms = 0;
    reply_ok();
    return;
  }
  if(starts_with(tokens[0], "INIT"))  { initialize();             return; }
  if(starts_with(tokens[0], "AUTO_PUSH")) { auto_push();          return; }

  if(g_auto_push_state != AUTO_PUSH_IDLE){
    reply_err(F("auto_push_busy"));
    return;
  }

  if(starts_with(tokens[0], "EN")){
    if(nt < 2){ reply_err(F("EN needs 0/1")); return; }
    g_enabled = (atoi(tokens[1]) != 0);
    if(!g_enabled) emergency_stop_all(false);
    reply_ok();
    return;
  }

  // ---- DC 直流电机 ----
  if(starts_with(tokens[0], "DC")){
    if(nt < 2){ reply_err(F("DC needs duty")); return; }
    if(tokens[1][0] == '0'){
      dc_stop();
      reply_ok();
      return;
    }
    int duty = atoi(tokens[1]);
    if(duty > 255) duty = 255;
    if(duty < -255) duty = -255;
    dc_set_speed((int16_t)duty);
    reply_ok();
    return;
  }

  // ---- 步进 MOVE ----
  if(starts_with(tokens[0], "MOVE")){
    if(nt < 3){ reply_err(F("MOVE needs args")); return; }

    char axis_name = 0;
    if(!extract_axis_from_token(tokens[1], axis_name)){
      reply_err(F("bad axis (X/Y/Z)"));
      return;
    }
    long     steps = atol(tokens[2]);
    uint16_t rate  = 800;
    float    aup   = 600.0f;
    float    adown = 600.0f;

    for(uint8_t i=3; i+1<nt; i++){
      char c0 = toupper(tokens[i][0]);
      char c1 = tokens[i][1] ? toupper(tokens[i][1]) : 0;
      if(c0=='F'){
        rate = (uint16_t)atoi(tokens[i+1]);
      } else if(c0=='A' && c1==0){
        float a = (float)atol(tokens[i+1]);
        aup = a; adown = a;
      } else if(c0=='A' && c1=='U'){
        aup = (float)atol(tokens[i+1]);
      } else if(c0=='A' && c1=='D'){
        adown = (float)atol(tokens[i+1]);
      }
    }

    Axis* ax = find_axis(axis_name);
    if(!ax){ reply_err(F("axis_not_found")); return; }

    if(rate < MIN_STEP_RATE) rate = MIN_STEP_RATE;
    if(rate > MAX_STEP_RATE) rate = MAX_STEP_RATE;
    if(aup   < 1.0f) aup   = 1.0f;
    if(adown < 1.0f) adown = 1.0f;

    if(start_move(ax, steps, rate, aup, adown)){
      reply_ok();
    }
    return;
  }

  // ---- Block-wise MOVE ----
  if(starts_with(tokens[0], "BM")){
    if(nt < 3){ reply_err(F("MOVE needs args")); return; }

    char axis_name = 0;
    if(!extract_axis_from_token(tokens[1], axis_name)){
      reply_err(F("bad axis (X/Y/Z)"));
      return;
    }
    long     steps = atol(tokens[2]) * ((axis_name == 'Z') ? (400) : (638));
    uint16_t rate  = 800;
    float    aup   = 600.0f;
    float    adown = 600.0f;

    for(uint8_t i=3; i+1<nt; i++){
      char c0 = toupper(tokens[i][0]);
      char c1 = tokens[i][1] ? toupper(tokens[i][1]) : 0;
      if(c0=='F'){
        rate = (uint16_t)atoi(tokens[i+1]);
      } else if(c0=='A' && c1==0){
        float a = (float)atol(tokens[i+1]);
        aup = a; adown = a;
      } else if(c0=='A' && c1=='U'){
        aup = (float)atol(tokens[i+1]);
      } else if(c0=='A' && c1=='D'){
        adown = (float)atol(tokens[i+1]);
      }
    }

    Axis* ax = find_axis(axis_name);
    if(!ax){ reply_err(F("axis_not_found")); return; }

    if(rate < MIN_STEP_RATE) rate = MIN_STEP_RATE;
    if(rate > MAX_STEP_RATE) rate = MAX_STEP_RATE;
    if(aup   < 1.0f) aup   = 1.0f;
    if(adown < 1.0f) adown = 1.0f;

    if(start_move(ax, steps, rate, aup, adown)){
      if(axis_name == 'Z'){
        g_bm_z_done_pending = true;
      }
      reply_ok();
    }
    return;
  }

  reply_err(F("unknown_cmd"));
}

// ============================================================================
// setup / loop
// ============================================================================
void setup(){
  // D13 被用作 Z_DIR，不能再作为心跳 LED 控制。

  // 步进 STEP/DIR
  for(uint8_t i=0; i<NUM_AXES; i++){
    pinMode(ACTIVE_AXES[i]->pin_step, OUTPUT);
    pinMode(ACTIVE_AXES[i]->pin_dir,  OUTPUT);
    step_idle(ACTIVE_AXES[i]);
    write_dir(ACTIVE_AXES[i], false);
  }

  // 限位（NPN 开漏 + 内部上拉）
  pinMode(PIN_LSX1, INPUT_PULLUP);
  pinMode(PIN_LSX2, INPUT_PULLUP);
  pinMode(PIN_LSY1, INPUT_PULLUP);
  pinMode(PIN_LSY2, INPUT_PULLUP);
  pinMode(A0, INPUT);

  // H 桥 / BTS7960
  // R_EN / L_EN 已接 VCC，不占用 Arduino 引脚。
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);
  digitalWrite(PIN_RPWM, LOW);
  digitalWrite(PIN_LPWM, LOW);

  Serial.begin(SERIAL_BAUD);
  Serial.println(F("=== Custom Board Motion Firmware ==="));
  Serial.println(F("Type HELP for commands."));
  cmd_limits();   // 上电自检：打印一次限位状态

  timer1_start_step_scheduler();
  timer2_start_dc_pwm();
}

void loop(){
  // D13 是 Z_DIR，不能闪烁板载 LED；只执行运动更新。
  update_motion();

  // 完成事件上报（含 LIMIT 标记）
  static bool was_active[NUM_AXES] = {false, false, false};
  for(uint8_t i=0; i<NUM_AXES; i++){
    bool now_active = ACTIVE_AXES[i]->active;
    bool bm_z_done_ready = (ACTIVE_AXES[i] == &AXIS_Z && g_bm_z_done_pending && !now_active);
    if((was_active[i] && !now_active) || bm_z_done_ready){
      Serial.print(F("DONE "));
      Serial.print(ACTIVE_AXES[i]->name);
      if(ACTIVE_AXES[i]->limit_hit){
        Serial.print(F(" LIMIT"));
        ACTIVE_AXES[i]->limit_hit = false;
      }
      Serial.println();
      if(ACTIVE_AXES[i] == &AXIS_Z){
        g_bm_z_done_pending = false;
      }
    }
    was_active[i] = now_active;
  }

  update_initialize();
  update_auto_push();
  update_a0_test();

  // 串口接收
  while(Serial.available()){
    char c = (char)Serial.read();
    if(c == '\n'){
      linebuf[linepos] = 0;
      handle_line(linebuf);
      linepos = 0;
    } else if(c != '\r'){
      if(linepos < sizeof(linebuf)-1){
        linebuf[linepos++] = c;
      } else {
        linepos = 0;
        reply_err(F("line_too_long"));
      }
    }
  }
}
