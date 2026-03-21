"""
Pico 1 — GUN CONTROLLER
========================
Responsibilities:
  1. Read BMI160 IMU → send AIM:yaw,pitch to laptop over USB serial
  2. Read trigger button → send SHOOT to laptop + fire IR LED + signal robot via nRF24
  3. Read reload button  → send RELOAD to laptop
  4. Read joystick      → send robot movement commands to Pico 2 via nRF24
  5. Forward IMU aim to Pico 2 via nRF24 → robot head servo tracks gun direction
  6. Listen for HIT reply from Pico 2 via nRF24 → send HIT to laptop

nRF24 packet sent to Pico 2 every ~20ms (7 bytes):
  [joy_x: int8]  [-100..100]
  [joy_y: int8]  [-100..100]
  [yaw:  int16]  angle × 10  e.g. 12.5° → 125
  [pitch:int16]  angle × 10
  [flags: uint8] bit0=SHOOT  bit1=RELOAD

nRF24 packet received from Pico 2 (1 byte):
  [flags: uint8] bit0=HIT

USB serial to laptop (115200 baud), newline-terminated:
  AIM:yaw,pitch   ~50 Hz
  SHOOT           on trigger
  RELOAD          on reload button
  HIT             when robot reports it was hit by the IR gun

Wiring
------
BMI160        SDA→GP0  SCL→GP1  VCC→3.3V  GND→GND
Joystick X    →GP26 (ADC0)
Joystick Y    →GP27 (ADC1)
Joystick btn  →GP28  (other leg to GND, internal pull-up)
Trigger btn   →GP14  (other leg to GND, internal pull-up)
Reload btn    →GP15  (other leg to GND, internal pull-up)
IR LED        →GP16 → 100Ω → GND
nRF24 SCK→GP2  MOSI→GP3  MISO→GP4  CSN→GP5  CE→GP6  VCC→3.3V  GND→GND
"""

import machine, time, struct, sys, math
from machine import I2C, Pin, SPI, ADC

# ── PIN CONFIG ────────────────────────────────────────────────────────────────
TRIGGER_PIN  = 14
RELOAD_PIN   = 15
IR_LED_PIN   = 16
JOY_X_PIN    = 26   # ADC0
JOY_Y_PIN    = 27   # ADC1
JOY_BTN_PIN  = 28   # optional joystick button
IMU_SDA      = 0
IMU_SCL      = 1
BMI160_ADDR  = 0x68  # change to 0x69 if SDO is pulled HIGH
NRF_SCK, NRF_MOSI, NRF_MISO, NRF_CSN, NRF_CE = 2, 3, 4, 5, 6

# ── ADDRESSES ────────────────────────────────────────────────────────────────
ADDR_ROBOT = b'\xD2\xF0\xF0\xF0\xF0'  # Pico 2 RX address (gun sends here)
ADDR_GUN   = b'\xE1\xF0\xF0\xF0\xF0'  # Pico 1 RX address (robot replies here)

# ── TUNING ───────────────────────────────────────────────────────────────────
SEND_HZ       = 50      # AIM + nRF24 update rate
DEBOUNCE_MS   = 60
YAW_DEADZONE  = 0.8     # deg/s — kills gyro drift
PITCH_DEADZONE= 0.5
JOY_DEADZONE  = 3000    # ADC units (0–65535) — ignores small stick movements
JOY_CENTER    = 32768   # ADC midpoint for a resting joystick


# ═══════════════════════════════════════════════════════════════════════════════
#  BMI160 IMU DRIVER
# ═══════════════════════════════════════════════════════════════════════════════
class BMI160:
    def __init__(self, i2c, addr=0x68):
        self.i2c  = i2c
        self.addr = addr
        self._w(0x7E, 0xB6); time.sleep_ms(150)  # soft reset
        self._w(0x7E, 0x11); time.sleep_ms(50)   # accel normal
        self._w(0x7E, 0x15); time.sleep_ms(50)   # gyro normal
        self._w(0x43, 0x02)                        # gyro ±500 dps
        self._gyro_scale = 1.0 / 65.6

    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val]))

    def _r(self, reg, n):
        return self.i2c.readfrom_mem(self.addr, reg, n)

    def gyro(self):
        gx, gy, gz = struct.unpack_from('<hhh', self._r(0x0C, 6))
        s = self._gyro_scale
        return gx*s, gy*s, gz*s

    def accel(self):
        ax, ay, az = struct.unpack_from('<hhh', self._r(0x12, 6))
        s = 1.0 / 16384.0
        return ax*s, ay*s, az*s


# ═══════════════════════════════════════════════════════════════════════════════
#  nRF24L01+ DRIVER  (PTX primary, brief RX window to receive HIT from robot)
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
        self.csn.value(0); self.spi.write(bytes([reg])); v=self.spi.read(1)[0]; self.csn.value(1)
        return v

    def _cmd(self, cmd):
        self.csn.value(0); self.spi.write(bytes([cmd])); self.csn.value(1)

    def _init_ptx(self):
        time.sleep_ms(5)
        self._w(0x00, 0x0E)           # CONFIG: CRC 2B, PWR_UP, PTX
        self._w(0x01, 0x01)           # EN_AA: auto-ack pipe 0
        self._w(0x02, 0x03)           # EN_RXADDR: pipe 0 + pipe 1
        self._w(0x03, 0x03)           # AW: 5-byte address
        self._w(0x04, 0x1F)           # SETUP_RETR: 500µs, 15 retries
        self._w(0x05, 100)            # RF_CH: channel 100
        self._w(0x06, 0x0F)           # RF_SETUP: 2Mbps, max power
        self._wb(0x10, ADDR_ROBOT)    # TX_ADDR: send to robot
        self._wb(0x0A, ADDR_ROBOT)    # RX_ADDR_P0: for auto-ack
        self._wb(0x0B, ADDR_GUN)      # RX_ADDR_P1: receive HIT replies
        self._w(0x11, 7)              # RX_PW_P0: 7-byte payload
        self._w(0x12, 1)              # RX_PW_P1: 1-byte HIT reply
        self._cmd(0xE1)               # flush TX FIFO
        self._cmd(0xE2)               # flush RX FIFO
        self._w(0x07, 0x70)           # clear interrupt flags

    def send(self, data):
        """Send up to 32-byte packet. Returns True on success."""
        self._cmd(0xE1)               # flush TX
        self._w(0x07, 0x70)          # clear flags
        self.csn.value(0); self.spi.write(b'\xA0' + data); self.csn.value(1)
        self.ce.value(1); time.sleep_us(15); self.ce.value(0)
        # Wait for TX_DS (success) or MAX_RT (fail), timeout 5ms
        deadline = time.ticks_add(time.ticks_ms(), 5)
        while time.ticks_diff(deadline, time.ticks_ms()) > 0:
            status = self._r(0x07)
            if status & 0x20: return True   # TX_DS
            if status & 0x10: self._cmd(0xE1); return False  # MAX_RT
        return False

    def check_hit(self):
        """Briefly enter RX mode to check for a HIT reply from the robot."""
        # Switch to RX
        self._w(0x00, 0x0F)           # PRX mode
        self.ce.value(1)
        time.sleep_us(200)            # listen for 200µs
        hit = False
        status = self._r(0x07)
        if status & 0x40:             # RX_DR: data ready
            self.csn.value(0); self.spi.write(b'\x61'); payload=self.spi.read(1)[0]; self.csn.value(1)
            if payload & 0x01:        # bit 0 = HIT flag
                hit = True
            self._w(0x07, 0x40)       # clear RX_DR
        # Switch back to PTX
        self.ce.value(0)
        self._w(0x00, 0x0E)
        return hit


# ═══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INIT
# ═══════════════════════════════════════════════════════════════════════════════
i2c     = I2C(0, sda=Pin(IMU_SDA), scl=Pin(IMU_SCL), freq=400_000)
imu     = BMI160(i2c, BMI160_ADDR)

joy_x   = ADC(Pin(JOY_X_PIN))
joy_y   = ADC(Pin(JOY_Y_PIN))
joy_btn = Pin(JOY_BTN_PIN, Pin.IN, Pin.PULL_UP)
trigger = Pin(TRIGGER_PIN,  Pin.IN, Pin.PULL_UP)
reload_btn = Pin(RELOAD_PIN, Pin.IN, Pin.PULL_UP)
ir_led  = Pin(IR_LED_PIN, Pin.OUT); ir_led.value(0)

spi = SPI(0, baudrate=4_000_000, sck=Pin(NRF_SCK),
          mosi=Pin(NRF_MOSI), miso=Pin(NRF_MISO))
nrf = NRF24(spi, Pin(NRF_CSN, Pin.OUT), Pin(NRF_CE, Pin.OUT))

# ═══════════════════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════════════════
yaw, pitch = 0.0, 0.0
last_trigger_state = 1
last_reload_state  = 1
last_trig_ms = last_rel_ms = last_send_ms = 0
last_loop_ms = time.ticks_ms()
INTERVAL_MS  = 1000 // SEND_HZ

sys.stdout.write("PICO1_READY\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def read_joystick():
    """Return (x, y) in -100..100 with deadzone applied."""
    raw_x = joy_x.read_u16()
    raw_y = joy_y.read_u16()
    dx = raw_x - JOY_CENTER
    dy = raw_y - JOY_CENTER
    if abs(dx) < JOY_DEADZONE: dx = 0
    if abs(dy) < JOY_DEADZONE: dy = 0
    x = max(-100, min(100, int(dx / 327)))   # 32767/100 ≈ 327
    y = max(-100, min(100, int(dy / 327)))
    return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
while True:
    now = time.ticks_ms()
    dt  = max(0.001, time.ticks_diff(now, last_loop_ms) / 1000.0)
    last_loop_ms = now

    # ── IMU ──────────────────────────────────────────────────────────────────
    try:
        gx, gy, gz = imu.gyro()
        ax, ay, az = imu.accel()
    except Exception:
        time.sleep_ms(10); continue

    if abs(gz) > YAW_DEADZONE:
        yaw += gz * dt
    try:
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
    except:
        accel_pitch = pitch
    pitch = 0.02 * accel_pitch + 0.98 * (pitch + gy * dt)
    yaw   = max(-180.0, min(180.0, yaw))
    pitch = max(-90.0,  min(90.0,  pitch))

    # ── BUTTONS ──────────────────────────────────────────────────────────────
    shoot_flag   = False
    reload_flag  = False

    trig = trigger.value()
    if trig == 0 and last_trigger_state == 1:
        if time.ticks_diff(now, last_trig_ms) > DEBOUNCE_MS:
            sys.stdout.write("SHOOT\n")
            ir_led.value(1); time.sleep_ms(25); ir_led.value(0)
            shoot_flag = True
            last_trig_ms = now
    last_trigger_state = trig

    rel = reload_btn.value()
    if rel == 0 and last_reload_state == 1:
        if time.ticks_diff(now, last_rel_ms) > DEBOUNCE_MS:
            sys.stdout.write("RELOAD\n")
            # Hold reload + trigger → re-zero aim
            if trigger.value() == 0:
                yaw = 0.0; pitch = 0.0
                sys.stdout.write("AIM:0.0,0.0\n")
            reload_flag = True
            last_rel_ms = now
    last_reload_state = rel

    # ── SEND TO LAPTOP + ROBOT ────────────────────────────────────────────────
    if time.ticks_diff(now, last_send_ms) >= INTERVAL_MS:
        # USB serial → laptop
        sys.stdout.write("AIM:{:.1f},{:.1f}\n".format(yaw, pitch))

        # nRF24 → robot (joy + aim + flags)
        jx, jy = read_joystick()
        flags  = (0x01 if shoot_flag else 0) | (0x02 if reload_flag else 0)
        packet = struct.pack('<bbhhB',
                             jx, jy,
                             int(yaw * 10),
                             int(pitch * 10),
                             flags)
        nrf.send(packet)

        # Check for HIT reply from robot (brief RX window after TX)
        if nrf.check_hit():
            sys.stdout.write("HIT\n")

        last_send_ms = now

    # Sleep remainder
    elapsed = time.ticks_diff(time.ticks_ms(), now)
    rem = INTERVAL_MS - elapsed
    if rem > 0:
        time.sleep_ms(rem)
