# pico1_gun.py — Gun controller firmware
# Upload this file + bmi160.py + nrf24l01.py to the Gun Pico
#
# Wiring
# ─────────────────────────────────────────────────────
# BMI160    SDA → GP4   SCL → GP5   VCC → 3.3V   GND → GND
# Joystick1 X   → GP26 (steer left/right → robot movement)
# Joystick1 Y   → GP27 (throttle fwd/back → robot movement)
# Joystick2 Y   → GP28 (pull to fire)
# Reload btn    → GP15  (other leg to GND, internal pull-up)
# IR LED        → GP14 → 100Ω → GND  (not yet wired)
#
# nRF24L01+ (SPI0):
#   GP18 → SCK   GP19 → MOSI   GP16 → MISO
#   GP17 → CE    GP20 → CSN
#   3V3  → VCC (add 10µF cap!)   GND → GND
#
# USB serial to laptop (115200 baud) — format understood by video streaming.py:
#   AIM:yaw,roll     ~50 Hz  (yaw=left/right, roll=barrel up/down)
#   SHOOT            when fire trigger deflected past threshold
#   RELOAD           when reload button pressed
#
# nRF24 packet to Tank Pico (16 bytes, little-endian):
#   [yaw:      float]
#   [roll:     float]
#   [steer:    int8]   -100..100
#   [throttle: int8]   -100..100
#   [fire:     uint8]  0 or 1
#   [pad:      1 byte]

import struct
import sys
from machine import Pin, I2C, ADC, SPI
from time import sleep_ms, ticks_ms, ticks_diff
from bmi160 import BMI160
from nrf24l01 import NRF24L01

# ── PIN CONFIG ────────────────────────────────────────────────────────────────
I2C_SDA  = 4
I2C_SCL  = 5
JOY1_X   = 26   # steer  (robot left/right)
JOY1_Y   = 27   # throttle (robot fwd/back)
JOY2_Y   = 28   # fire trigger
JOY2_SW  = 22   # fire joystick center button (press to zero aim)
RELOAD_PIN = 15
IR_LED_PIN = 14  # moved from GP16 (nRF24 MISO uses GP16)

# nRF24 on SPI0 (matching friend's tested wiring)
NRF_SCK  = 18
NRF_MOSI = 19
NRF_MISO = 16
NRF_CSN  = 20
NRF_CE   = 17

# ── RADIO CONFIG (must match tank_main) ─────────────────────────────────────
CHANNEL = 76
PIPE    = b"\xe1\xf0\xf0\xf0\xf0"

# ── TUNING ────────────────────────────────────────────────────────────────────
SEND_HZ        = 50
FIRE_THRESHOLD = 25   # joystick deflection % to count as firing
DEBOUNCE_MS    = 60
JOY_DEADZONE   = 4000 # ADC units

# ── RELOAD GESTURE (flick gun up-and-down) ──────────────────────────────────
FLICK_THRESHOLD    = 120   # dps — GX must exceed this to start a flick
FLICK_REVERSE_THR  = 80    # dps — GX must go below -this to confirm reversal
FLICK_WINDOW_MS    = 400   # ms  — reversal must happen within this window
RELOAD_COOLDOWN_MS = 1500  # ms  — ignore gestures for this long after reload

# ── PACKET FORMAT (16 bytes, must match tank) ────────────────────────────────
PACK_FMT = '<ffbbBx'   # yaw(f) + roll(f) + steer(b) + throttle(b) + fire(B) + pad


# ═══════════════════════════════════════════════════════════════════════════════
#  JOYSTICK HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate_joysticks(samples=50):
    print("Calibrating joysticks... hands off!")
    sleep_ms(500)
    s1x = s1y = s2y = 0
    for _ in range(samples):
        s1x += joy1_x.read_u16()
        s1y += joy1_y.read_u16()
        s2y += joy2_y.read_u16()
        sleep_ms(10)
    center = {'j1x': s1x // samples, 'j1y': s1y // samples, 'j2y': s2y // samples}
    print(f"Centers: J1=({center['j1x']}, {center['j1y']}) J2y={center['j2y']}")
    return center

def read_joystick(adc, center, deadzone=JOY_DEADZONE):
    raw    = adc.read_u16()
    offset = raw - center
    if abs(offset) < deadzone:
        return 0
    if offset > 0:
        val = (offset - deadzone) / (32767 - deadzone) * 100
    else:
        val = (offset + deadzone) / (32767 - deadzone) * 100
    return max(-100, min(100, int(val)))


# ═══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== GUN CONTROLLER STARTUP ===\n")

i2c = I2C(0, sda=Pin(I2C_SDA), scl=Pin(I2C_SCL), freq=400_000)
devices = i2c.scan()
print(f"I2C devices: {[hex(d) for d in devices]}")
if 0x68 not in devices and 0x69 not in devices:
    print("ERROR: BMI160 not found!"); raise SystemExit

addr = 0x68 if 0x68 in devices else 0x69
imu  = BMI160(i2c, addr=addr)
imu.init()

joy1_x = ADC(Pin(JOY1_X))
joy1_y = ADC(Pin(JOY1_Y))
joy2_y  = ADC(Pin(JOY2_Y))
joy2_sw = Pin(JOY2_SW, Pin.IN, Pin.PULL_UP)
reload_btn = Pin(RELOAD_PIN, Pin.IN, Pin.PULL_UP)
ir_led     = Pin(IR_LED_PIN, Pin.OUT); ir_led.value(0)

# Radio
print("Initialising nRF24L01+...")
spi = SPI(0, sck=Pin(NRF_SCK), mosi=Pin(NRF_MOSI), miso=Pin(NRF_MISO))
nrf = NRF24L01(spi, Pin(NRF_CSN, Pin.OUT), Pin(NRF_CE, Pin.OUT),
               channel=CHANNEL, payload_size=16)
nrf.open_tx_pipe(PIPE)
nrf.stop_listening()
print("Radio OK")

# Calibrate (keep gun still, hands off joysticks)
print("\nKeep gun STILL and hands OFF joysticks!")
sleep_ms(1000)
imu.calibrate(samples=500, delay_ms=10)
joy_center = calibrate_joysticks()

print("\n=== READY — streaming to laptop ===")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════════════════
last_fire_state   = False
last_push_state   = False
last_sw_state     = 1       # joystick button (active low)
flick_state       = 0       # 0=idle, 1=upstroke detected
flick_start_ms    = 0
last_reload_ms    = 0
INTERVAL_MS       = 1000 // SEND_HZ
tx_ok   = 0
tx_fail = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while True:
    t0 = ticks_ms()

    # ── IMU ──────────────────────────────────────────────────────────────────
    yaw, roll, gz, gx = imu.update()

    # ── JOYSTICKS ────────────────────────────────────────────────────────────
    steer    = read_joystick(joy1_y, joy_center['j1y'])
    throttle = read_joystick(joy1_x, joy_center['j1x'])
    fire_raw = read_joystick(joy2_y, joy_center['j2y'])
    fire_pull = fire_raw > FIRE_THRESHOLD     # pull toward you = fire
    fire_push = fire_raw < -FIRE_THRESHOLD    # push away = centre crosshair

    # ── SHOOT (rising edge of pull trigger) ──────────────────────────────────
    shoot_flag = False
    if fire_pull and not last_fire_state:
        sys.stdout.write("SHOOT\n")
        ir_led.value(1); sleep_ms(25); ir_led.value(0)
        shoot_flag = True
    last_fire_state = fire_pull

    # ── ZERO AIM (push joystick away) ────────────────────────────────────────
    if fire_push and not last_push_state:
        imu.reset_yaw()
        sys.stdout.write("ZERO\n")
    last_push_state = fire_push

    # ── ZERO AIM (joystick center button press — backup) ─────────────────────
    sw = joy2_sw.value()
    if sw == 0 and last_sw_state == 1:
        imu.reset_yaw()
        sys.stdout.write("ZERO\n")
    last_sw_state = sw

    # ── RELOAD (flick gun up-and-down gesture) ───────────────────────────────
    now_ms = ticks_ms()
    cooldown_ok = ticks_diff(now_ms, last_reload_ms) > RELOAD_COOLDOWN_MS
    if flick_state == 0:
        if gx > FLICK_THRESHOLD and cooldown_ok:
            flick_state = 1
            flick_start_ms = now_ms
    elif flick_state == 1:
        if gx < -FLICK_REVERSE_THR:
            sys.stdout.write("RELOAD\n")
            flick_state = 0
            last_reload_ms = now_ms
        elif ticks_diff(now_ms, flick_start_ms) > FLICK_WINDOW_MS:
            flick_state = 0

    # ── SEND TO LAPTOP ───────────────────────────────────────────────────────
    sys.stdout.write("AIM:{:.1f},{:.1f}\n".format(yaw, roll))
    sys.stdout.write("JOY:{},{}\n".format(steer, throttle))

    # ── SEND TO TANK (nRF24) ─────────────────────────────────────────────────
    fire = 1 if shoot_flag else 0
    payload = struct.pack(PACK_FMT, yaw, roll, steer, throttle, fire)
    payload += b'\x00' * (16 - len(payload))

    try:
        nrf.send(payload)
        tx_ok += 1
    except OSError:
        tx_fail += 1

    # ── MAINTAIN 50 Hz ───────────────────────────────────────────────────────
    elapsed = ticks_diff(ticks_ms(), t0)
    rem = INTERVAL_MS - elapsed
    if rem > 0:
        sleep_ms(rem)
