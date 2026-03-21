"""
Pico 2 — ROBOT CONTROLLER
==========================
Responsibilities:
  1. Receive nRF24 packets from Pico 1 (gun)
  2. Joystick data  → drive DC motors (robot moves)
  3. IMU aim data   → pan/tilt servos rotate robot head to match gun direction
  4. SHOOT flag     → fire servo launcher (robot shoots back in real life)
  5. IR receiver    → detects when player's gun hits the robot →
                       sets HIT flag in next nRF24 reply to Pico 1

nRF24 packet received from Pico 1 (7 bytes):
  [joy_x: int8]   [-100..100]  left/right
  [joy_y: int8]   [-100..100]  forward/backward
  [yaw:   int16]  angle × 10
  [pitch: int16]  angle × 10
  [flags: uint8]  bit0=SHOOT  bit1=RELOAD

nRF24 reply to Pico 1 (1 byte, via PTX after receiving):
  [flags: uint8]  bit0=HIT

Wiring
------
nRF24 SCK→GP2  MOSI→GP3  MISO→GP4  CSN→GP5  CE→GP6  VCC→3.3V  GND→GND

IR Receiver (TSOP38238 or similar 38kHz)
  OUT→GP16   VCC→3.3V   GND→GND

DC Motors via L298N
  ENA (left  PWM)  → GP10     ENB (right PWM)  → GP13
  IN1 (left  fwd)  → GP11     IN3 (right fwd)  → GP14
  IN2 (left  bck)  → GP12     IN4 (right bck)  → GP15

Head pan servo  (yaw)   → GP20
Head tilt servo (pitch) → GP21
Launcher servo          → GP22
"""

import machine, time, struct
from machine import Pin, SPI, PWM

# ── PIN CONFIG ────────────────────────────────────────────────────────────────
IR_RECV_PIN = 16
NRF_SCK, NRF_MOSI, NRF_MISO, NRF_CSN, NRF_CE = 2, 3, 4, 5, 6
MOTOR_ENA, MOTOR_IN1, MOTOR_IN2 = 10, 11, 12   # left motor
MOTOR_ENB, MOTOR_IN3, MOTOR_IN4 = 13, 14, 15   # right motor
SERVO_PAN   = 20   # head left/right (yaw)
SERVO_TILT  = 21   # head up/down   (pitch)
SERVO_LAUNCH= 22   # launcher

# ── ADDRESSES (must match pico1_gun.py) ──────────────────────────────────────
ADDR_ROBOT = b'\xD2\xF0\xF0\xF0\xF0'  # this Pico's RX address
ADDR_GUN   = b'\xE1\xF0\xF0\xF0\xF0'  # Pico 1's RX address (for HIT replies)

# ── TUNING ───────────────────────────────────────────────────────────────────
MOTOR_SPEED  = 55000   # 0–65535 (tune for your motors)
IR_COOLDOWN  = 500     # ms — ignore duplicate IR triggers

# Head servo range: map gun yaw ±90° → servo 500–2500µs
# Adjust these limits if your servo has a different range
SERVO_MIN_US  = 500
SERVO_MAX_US  = 2500
SERVO_MID_US  = 1500   # neutral / straight ahead

# Launcher
LAUNCHER_FIRE_US    = 2000
LAUNCHER_NEUTRAL_US = 1500
LAUNCHER_COOLDOWN   = 3000  # ms between shots


# ═══════════════════════════════════════════════════════════════════════════════
#  nRF24L01+ DRIVER  (PRX primary, brief PTX window to send HIT reply)
# ═══════════════════════════════════════════════════════════════════════════════
class NRF24:
    def __init__(self, spi, csn, ce):
        self.spi = spi
        self.csn = csn
        self.ce  = ce
        self.ce.value(0)
        self._init_prx()

    def _w(self, reg, val):
        self.csn.value(0); self.spi.write(bytes([0x20|reg, val])); self.csn.value(1)

    def _wb(self, reg, data):
        self.csn.value(0); self.spi.write(bytes([0x20|reg]) + data); self.csn.value(1)

    def _r(self, reg):
        self.csn.value(0); self.spi.write(bytes([reg])); v=self.spi.read(1)[0]; self.csn.value(1)
        return v

    def _cmd(self, c):
        self.csn.value(0); self.spi.write(bytes([c])); self.csn.value(1)

    def _init_prx(self):
        time.sleep_ms(5)
        self._w(0x00, 0x0F)           # CONFIG: CRC 2B, PWR_UP, PRX
        self._w(0x01, 0x01)           # EN_AA: auto-ack pipe 0
        self._w(0x02, 0x01)           # EN_RXADDR: pipe 0
        self._w(0x03, 0x03)           # AW: 5-byte address
        self._w(0x05, 100)            # RF_CH: channel 100
        self._w(0x06, 0x0F)           # RF_SETUP: 2Mbps, max power
        self._wb(0x0A, ADDR_ROBOT)    # RX_ADDR_P0
        self._w(0x11, 7)              # RX_PW_P0: 7-byte payload
        self._cmd(0xE1); self._cmd(0xE2)  # flush FIFOs
        self._w(0x07, 0x70)           # clear flags
        self.ce.value(1)              # start listening

    def available(self):
        return bool(self._r(0x07) & 0x40)   # RX_DR

    def recv(self):
        """Read 7-byte packet."""
        self.csn.value(0); self.spi.write(b'\x61'); data=self.spi.read(7); self.csn.value(1)
        self._w(0x07, 0x40)
        return data

    def send_hit(self):
        """Switch to PTX, send 1-byte HIT packet to Pico 1, switch back to PRX."""
        self.ce.value(0)
        self._w(0x00, 0x0E)           # PTX mode
        self._wb(0x10, ADDR_GUN)      # TX address = gun
        self._wb(0x0A, ADDR_GUN)      # pipe 0 addr for auto-ack
        self._w(0x11, 1)              # TX payload = 1 byte
        self._cmd(0xE1)               # flush TX FIFO
        self._w(0x07, 0x70)
        self.csn.value(0); self.spi.write(b'\xA0\x01'); self.csn.value(1)  # payload = 0x01 (HIT)
        self.ce.value(1); time.sleep_us(15); self.ce.value(0)
        time.sleep_ms(4)              # wait for transmission
        # Restore PRX
        self._w(0x00, 0x0F)
        self._wb(0x0A, ADDR_ROBOT)    # restore pipe 0 addr
        self._w(0x11, 7)
        self._cmd(0xE1); self._cmd(0xE2)
        self._w(0x07, 0x70)
        self.ce.value(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR DRIVER
# ═══════════════════════════════════════════════════════════════════════════════
class Motors:
    def __init__(self):
        self.ena = PWM(Pin(MOTOR_ENA)); self.ena.freq(1000)
        self.enb = PWM(Pin(MOTOR_ENB)); self.enb.freq(1000)
        self.in1 = Pin(MOTOR_IN1, Pin.OUT)
        self.in2 = Pin(MOTOR_IN2, Pin.OUT)
        self.in3 = Pin(MOTOR_IN3, Pin.OUT)
        self.in4 = Pin(MOTOR_IN4, Pin.OUT)
        self.stop()

    def drive(self, joy_x, joy_y):
        """
        joy_x: -100 (left) to +100 (right)
        joy_y: -100 (backward) to +100 (forward)
        Tank-style steering: each stick side controls one motor.
        """
        # Mix joystick into left/right motor speeds
        left  = joy_y + joy_x
        right = joy_y - joy_x
        left  = max(-100, min(100, left))
        right = max(-100, min(100, right))

        self._set_motor(self.in1, self.in2, self.ena, left)
        self._set_motor(self.in3, self.in4, self.enb, right)

    def _set_motor(self, in_a, in_b, en, speed):
        if speed > 0:
            in_a.value(1); in_b.value(0)
        elif speed < 0:
            in_a.value(0); in_b.value(1)
        else:
            in_a.value(0); in_b.value(0)
        duty = int(abs(speed) / 100 * MOTOR_SPEED)
        en.duty_u16(duty)

    def stop(self):
        self.ena.duty_u16(0); self.enb.duty_u16(0)
        self.in1.value(0); self.in2.value(0)
        self.in3.value(0); self.in4.value(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVO DRIVER
# ═══════════════════════════════════════════════════════════════════════════════
class Servo:
    def __init__(self, pin):
        self.pwm = PWM(Pin(pin)); self.pwm.freq(50)
        self.set_us(SERVO_MID_US)

    def set_us(self, us):
        us   = max(SERVO_MIN_US, min(SERVO_MAX_US, us))
        duty = int(us / 20000 * 65535)
        self.pwm.duty_u16(duty)

    def set_angle(self, angle, min_angle=-90, max_angle=90):
        """Map angle (degrees) to servo pulse width."""
        frac = (angle - min_angle) / (max_angle - min_angle)
        us   = SERVO_MIN_US + frac * (SERVO_MAX_US - SERVO_MIN_US)
        self.set_us(int(us))


# ═══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INIT
# ═══════════════════════════════════════════════════════════════════════════════
ir_recv  = Pin(IR_RECV_PIN, Pin.IN, Pin.PULL_UP)  # TSOP: LOW when IR received

spi = SPI(0, baudrate=4_000_000, sck=Pin(NRF_SCK),
          mosi=Pin(NRF_MOSI), miso=Pin(NRF_MISO))
nrf      = NRF24(spi, Pin(NRF_CSN, Pin.OUT), Pin(NRF_CE, Pin.OUT))
motors   = Motors()
pan      = Servo(SERVO_PAN)
tilt     = Servo(SERVO_TILT)
launcher = Servo(SERVO_LAUNCH)

# ── State ─────────────────────────────────────────────────────────────────────
last_ir_ms       = 0
last_launch_ms   = 0
pending_hit      = False   # set by IR, cleared after sending to Pico 1

print("PICO2_READY")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while True:
    now = time.ticks_ms()

    # ── IR HIT DETECTION ─────────────────────────────────────────────────────
    # TSOP output goes LOW when 38kHz IR is received
    if ir_recv.value() == 0:
        if time.ticks_diff(now, last_ir_ms) > IR_COOLDOWN:
            pending_hit = True
            last_ir_ms  = now
            print("IR_HIT")

    # ── SEND HIT REPLY TO GUN (and clear flag) ────────────────────────────────
    if pending_hit:
        nrf.send_hit()
        pending_hit = False

    # ── RECEIVE COMMAND PACKET FROM GUN ──────────────────────────────────────
    if nrf.available():
        try:
            data = nrf.recv()
            joy_x, joy_y, yaw_10, pitch_10, flags = struct.unpack('<bbhhB', data)
            yaw   = yaw_10   / 10.0
            pitch = pitch_10 / 10.0

            # Drive motors from joystick
            motors.drive(joy_x, joy_y)

            # Rotate head to match gun aim
            pan.set_angle(yaw,   min_angle=-90, max_angle=90)
            tilt.set_angle(pitch, min_angle=-45, max_angle=45)

            # Fire launcher if SHOOT flag and cooldown elapsed
            if (flags & 0x01) and time.ticks_diff(now, last_launch_ms) > LAUNCHER_COOLDOWN:
                launcher.set_us(LAUNCHER_FIRE_US)
                time.sleep_ms(400)
                launcher.set_us(LAUNCHER_NEUTRAL_US)
                last_launch_ms = now

        except Exception as e:
            print("RECV_ERR", e)

    time.sleep_ms(5)
