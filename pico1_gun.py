# pico1_gun.py — Gun controller firmware
# Upload this file AND bmi160.py to the Pico root
#
# Wiring (from working prototype)
# ─────────────────────────────────────────────────────
# BMI160    SDA → GP4   SCL → GP5   VCC → 3.3V   GND → GND
# Joystick1 X   → GP26 (steer left/right → robot movement)
# Joystick1 Y   → GP27 (throttle fwd/back → robot movement)
# Joystick2 Y   → GP28 (pull to fire)
# Reload btn    → GP15  (other leg to GND, internal pull-up)
# IR LED        → GP16 → 100Ω → GND
# nRF24  SCK→GP2  MOSI→GP3  MISO→GP4 ... wait, GP4 is I2C SDA
#        Use SPI1: SCK→GP10  MOSI→GP11  MISO→GP12  CSN→GP13  CE→GP14
#
# USB serial to laptop (115200 baud) — format understood by video streaming.py:
#   AIM:yaw,roll     ~50 Hz  (yaw=left/right, roll=barrel up/down)
#   SHOOT            when fire trigger deflected past threshold
#   RELOAD           when reload button pressed
#   HIT              when nRF24 receives hit notification from robot
#
# nRF24 packet to Pico 2 (7 bytes, little-endian):
#   [steer:  int8]   -100..100
#   [throttle:int8]  -100..100
#   [yaw:    int16]  degrees × 10
#   [roll:   int16]  degrees × 10
#   [flags:  uint8]  bit0=SHOOT

import json
import struct
import sys
from machine import Pin, I2C, ADC, SPI
from time import sleep_ms, ticks_ms, ticks_diff
from bmi160 import BMI160

# ── PIN CONFIG ────────────────────────────────────────────────────────────────
I2C_SDA  = 4
I2C_SCL  = 5
JOY1_X   = 26   # steer  (robot left/right)
JOY1_Y   = 27   # throttle (robot fwd/back)
JOY2_Y   = 28   # fire trigger
RELOAD_PIN = 15
IR_LED_PIN = 16

# nRF24 on SPI1 (avoids conflict with I2C on GP4)
NRF_SCK  = 10
NRF_MOSI = 11
NRF_MISO = 12
NRF_CSN  = 13
NRF_CE   = 9

# ── nRF24 ADDRESSES ───────────────────────────────────────────────────────────
ADDR_ROBOT = b'\xD2\xF0\xF0\xF0\xF0'
ADDR_GUN   = b'\xE1\xF0\xF0\xF0\xF0'

# ── TUNING ────────────────────────────────────────────────────────────────────
SEND_HZ        = 50
FIRE_THRESHOLD = 25   # joystick deflection % to count as firing
DEBOUNCE_MS    = 60
JOY_DEADZONE   = 4000 # ADC units — same as working prototype


# ═══════════════════════════════════════════════════════════════════════════════
#  nRF24L01+ DRIVER
# ═══════════════════════════════════════════════════════════════════════════════
class NRF24:
    def __init__(self, spi, csn, ce):
        self.spi = spi
        self.csn = csn
        self.ce  = ce
        self.ce.value(0)
        self._init_ptx()

    def _w(self, reg, val):
        self.csn.value(0); self.spi.write(bytes([0x20|reg, val])); self.csn.value(1)

    def _wb(self, reg, data):
        self.csn.value(0); self.spi.write(bytes([0x20|reg]) + data); self.csn.value(1)

    def _r(self, reg):
        self.csn.value(0); self.spi.write(bytes([reg])); v = self.spi.read(1)[0]; self.csn.value(1)
        return v

    def _flush(self):
        self.csn.value(0); self.spi.write(b'\xE1'); self.csn.value(1)  # flush TX
        self.csn.value(0); self.spi.write(b'\xE2'); self.csn.value(1)  # flush RX

    def _init_ptx(self):
        sleep_ms(5)
        self._w(0x00, 0x0E)           # CONFIG: CRC 2B, PWR_UP, PTX
        self._w(0x01, 0x01)           # EN_AA: auto-ack pipe 0
        self._w(0x02, 0x03)           # enable pipe 0 + pipe 1
        self._w(0x03, 0x03)           # 5-byte address
        self._w(0x04, 0x1F)           # 500µs delay, 15 retries
        self._w(0x05, 100)            # RF channel 100
        self._w(0x06, 0x0F)           # 2Mbps, max power
        self._wb(0x10, ADDR_ROBOT)    # TX address
        self._wb(0x0A, ADDR_ROBOT)    # pipe 0 (auto-ack)
        self._wb(0x0B, ADDR_GUN)      # pipe 1 (receive HIT replies)
        self._w(0x11, 7)              # pipe 0 payload = 7 bytes
        self._w(0x12, 1)              # pipe 1 payload = 1 byte (HIT)
        self._flush()
        self._w(0x07, 0x70)

    def send(self, data):
        self._flush()
        self._w(0x07, 0x70)
        self.csn.value(0); self.spi.write(b'\xA0' + data); self.csn.value(1)
        self.ce.value(1); sleep_ms(1); self.ce.value(0)
        deadline = ticks_ms() + 5
        while ticks_diff(deadline, ticks_ms()) > 0:
            s = self._r(0x07)
            if s & 0x20: return True
            if s & 0x10: self._flush(); return False
        return False

    def check_hit(self):
        """Briefly listen for HIT reply from robot."""
        self._w(0x00, 0x0F)       # PRX mode
        self.ce.value(1)
        sleep_ms(1)
        hit = False
        if self._r(0x07) & 0x40:
            self.csn.value(0); self.spi.write(b'\x61'); p = self.spi.read(1)[0]; self.csn.value(1)
            self._w(0x07, 0x40)
            hit = bool(p & 0x01)
        self.ce.value(0)
        self._w(0x00, 0x0E)       # back to PTX
        return hit


# ═══════════════════════════════════════════════════════════════════════════════
#  JOYSTICK HELPERS  (same logic as working prototype)
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
joy2_y = ADC(Pin(JOY2_Y))
reload_btn = Pin(RELOAD_PIN, Pin.IN, Pin.PULL_UP)
ir_led     = Pin(IR_LED_PIN, Pin.OUT); ir_led.value(0)

spi = SPI(1, baudrate=4_000_000, sck=Pin(NRF_SCK),
          mosi=Pin(NRF_MOSI), miso=Pin(NRF_MISO))
nrf = NRF24(spi, Pin(NRF_CSN, Pin.OUT), Pin(NRF_CE, Pin.OUT))

# Calibrate (keep gun still, hands off joysticks)
print("\nKeep gun STILL and hands OFF joysticks!")
sleep_ms(1000)
imu.calibrate(samples=500, delay_ms=10)
joy_center = calibrate_joysticks()

print("\n=== READY — streaming to laptop ===")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════════════════
last_reload_state = 1
last_rel_ms       = 0
last_fire_state   = False   # was fire active last loop
INTERVAL_MS       = 1000 // SEND_HZ


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while True:
    t0 = ticks_ms()

    # ── IMU ──────────────────────────────────────────────────────────────────
    yaw, roll, gz = imu.update()

    # ── JOYSTICKS ────────────────────────────────────────────────────────────
    steer    = read_joystick(joy1_x, joy_center['j1x'])
    throttle = read_joystick(joy1_y, joy_center['j1y'])
    fire_raw = read_joystick(joy2_y, joy_center['j2y'])
    fire_now = abs(fire_raw) > FIRE_THRESHOLD

    # ── SHOOT (rising edge of fire trigger) ──────────────────────────────────
    shoot_flag = False
    if fire_now and not last_fire_state:
        sys.stdout.write("SHOOT\n")
        ir_led.value(1); sleep_ms(25); ir_led.value(0)
        shoot_flag = True
    last_fire_state = fire_now

    # ── RELOAD BUTTON ────────────────────────────────────────────────────────
    reload_flag = False
    rel = reload_btn.value()
    if rel == 0 and last_reload_state == 1:
        if ticks_diff(ticks_ms(), last_rel_ms) > DEBOUNCE_MS:
            sys.stdout.write("RELOAD\n")
            # Hold reload + fire → re-zero yaw
            if fire_now:
                imu.reset_yaw()
                sys.stdout.write("AIM:0.0,0.0\n")
            reload_flag = True
            last_rel_ms = ticks_ms()
    last_reload_state = rel

    # ── SEND TO LAPTOP ───────────────────────────────────────────────────────
    sys.stdout.write("AIM:{:.1f},{:.1f}\n".format(yaw, roll))

    # ── SEND TO ROBOT (nRF24) ─────────────────────────────────────────────────
    flags  = 0x01 if shoot_flag else 0
    flags |= 0x02 if reload_flag else 0
    packet = struct.pack('<bbhhB',
                         steer, throttle,
                         int(yaw  * 10),
                         int(roll * 10),
                         flags)
    nrf.send(packet)

    # ── CHECK FOR HIT REPLY FROM ROBOT ───────────────────────────────────────
    if nrf.check_hit():
        sys.stdout.write("HIT\n")

    # ── MAINTAIN 50 Hz ───────────────────────────────────────────────────────
    elapsed = ticks_diff(ticks_ms(), t0)
    rem = INTERVAL_MS - elapsed
    if rem > 0:
        sleep_ms(rem)
