# pico2_robot.py — Tank receiver firmware with motor control
# Upload this + nrf24l01.py to the Tank Pico
#
# nRF24L01+ wiring (SPI0):
#   GP18 → SCK   GP19 → MOSI   GP16 → MISO
#   GP17 → CE    GP20 → CSN
#   3V3  → VCC (add 10µF cap!)   GND → GND
#
# L298N motor wiring:
#   GP2 → IN1, GP3 → IN2, GP6 → ENA (left motor)
#   GP7 → IN3, GP8 → IN4, GP9 → ENB (right motor)

import struct
from machine import Pin, SPI, PWM
from time import sleep_ms, ticks_ms, ticks_diff
from nrf24l01 import NRF24L01

# ── RADIO CONFIG (must match pico1_gun.py) ──────────────────────────────────
CHANNEL = 76
PIPE    = b"\xe1\xf0\xf0\xf0\xf0"

spi = SPI(0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
csn = Pin(20, Pin.OUT, value=1)
ce  = Pin(17, Pin.OUT, value=0)

# ── MOTOR SETUP ─────────────────────────────────────────────────────────────
in1 = Pin(2, Pin.OUT)
in2 = Pin(3, Pin.OUT)
ena = PWM(Pin(6))
ena.freq(1000)

in3 = Pin(7, Pin.OUT)
in4 = Pin(8, Pin.OUT)
enb = PWM(Pin(9))
enb.freq(1000)


def set_motor(in_a, in_b, en, speed):
    """Set a motor. speed: -100 to +100."""
    if speed > 0:
        in_a.high()
        in_b.low()
        en.duty_u16(int(speed / 100 * 65535))
    elif speed < 0:
        in_a.low()
        in_b.high()
        en.duty_u16(int(-speed / 100 * 65535))
    else:
        in_a.low()
        in_b.low()
        en.duty_u16(0)


def stop_motors():
    set_motor(in1, in2, ena, 0)
    set_motor(in3, in4, enb, 0)


def drive(throttle, steer):
    """Tank-style mixing: throttle (-100..100) + steer (-100..100)."""
    left = max(-100, min(100, throttle + steer))
    right = max(-100, min(100, throttle - steer))
    set_motor(in1, in2, ena, left)
    set_motor(in3, in4, enb, right)


# ── INIT ─────────────────────────────────────────────────────────────────────
print("\n=== TANK RECEIVER (RX) STARTUP ===\n")

print("Initialising nRF24L01+...")
nrf = NRF24L01(spi, csn, ce, channel=CHANNEL, payload_size=16)
nrf.open_rx_pipe(1, PIPE)
nrf.start_listening()
print("Radio OK — listening for gun controller...\n")

# Packet format (must match pico1_gun.py)
PACK_FMT = '<ffbbBx'

TIMEOUT_MS = 500
last_rx    = ticks_ms()
rx_count   = 0
connected  = False


# ── MAIN LOOP ────────────────────────────────────────────────────────────────
while True:
    if nrf.any():
        buf = nrf.recv()
        try:
            yaw, roll, steer, throttle, fire = struct.unpack(PACK_FMT, buf[:12])
        except Exception:
            continue

        steer    = max(-100, min(100, steer))
        throttle = max(-100, min(100, throttle))

        last_rx  = ticks_ms()
        rx_count += 1

        if not connected:
            print("Gun controller connected!")
            connected = True

        # Drive motors based on joystick input
        drive(throttle, steer)

        # Print status every 50 packets
        if rx_count % 50 == 0:
            print(f"RX#{rx_count}: yaw={yaw:.1f} roll={roll:.1f} steer={steer} thr={throttle} fire={fire}")

    # Safety: stop motors if no data received for TIMEOUT_MS
    if ticks_diff(ticks_ms(), last_rx) > TIMEOUT_MS:
        if connected:
            stop_motors()
            connected = False
            print("Signal lost — motors stopped")
            last_rx = ticks_ms()

    sleep_ms(5)
