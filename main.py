# main.py — Gun controller firmware for Pico 2
# Reads BMI160 (yaw/roll) + 2 joysticks, sends JSON over USB serial at ~50Hz
# Upload this + bmi160.py to the Pico

import json
from machine import Pin, I2C, ADC
from time import sleep_ms, ticks_ms, ticks_diff
from bmi160 import BMI160

# ============================================================
# PIN SETUP
# ============================================================

# I2C for BMI160
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=400000)

# Joystick 1: Tank movement (GP26 = X/steer, GP27 = Y/forward-back)
joy1_x = ADC(Pin(26))  # left/right steering
joy1_y = ADC(Pin(27))  # forward/backward

# Joystick 2: Fire trigger (GP28 = Y axis, any deflection = shoot)
joy2_y = ADC(Pin(28))  # pull to fire

# ============================================================
# JOYSTICK CALIBRATION
# ============================================================

def calibrate_joysticks(samples=50):
    """Read joystick center positions at startup (hands off!)"""
    print("Calibrating joysticks... hands off!")
    sleep_ms(500)
    
    s1x, s1y, s2y = 0, 0, 0
    for _ in range(samples):
        s1x += joy1_x.read_u16()
        s1y += joy1_y.read_u16()
        s2y += joy2_y.read_u16()
        sleep_ms(10)
    
    center = {
        'j1x': s1x // samples,
        'j1y': s1y // samples,
        'j2y': s2y // samples,
    }
    print(f"Joystick centers: J1=({center['j1x']}, {center['j1y']}) J2y={center['j2y']}")
    return center


def read_joystick(adc, center, deadzone=4000):
    """Read a joystick axis, return -100 to +100 with deadzone.
    0 = centered, negative = one direction, positive = the other.
    """
    raw = adc.read_u16()
    offset = raw - center
    
    # Apply deadzone
    if abs(offset) < deadzone:
        return 0
    
    # Normalize to -100..+100
    if offset > 0:
        val = (offset - deadzone) / (32767 - deadzone) * 100
    else:
        val = (offset + deadzone) / (32767 - deadzone) * 100
    
    # Clamp
    return max(-100, min(100, int(val)))


# ============================================================
# INIT
# ============================================================

print("\n=== GUN CONTROLLER STARTUP ===\n")

# Check I2C
devices = i2c.scan()
print(f"I2C devices: {[hex(d) for d in devices]}")
if 0x68 not in devices and 0x69 not in devices:
    print("ERROR: BMI160 not found!")
    raise SystemExit

addr = 0x68 if 0x68 in devices else 0x69
imu = BMI160(i2c, addr=addr)
imu.init()

# Calibrate everything (keep still, hands off joysticks!)
print("\nKeep sensor STILL and hands OFF joysticks!")
sleep_ms(1000)
imu.calibrate(samples=200, delay_ms=10)
joy_center = calibrate_joysticks()

# ============================================================
# MAIN LOOP — sends data at ~50Hz over USB serial
# ============================================================

print("\n=== STREAMING DATA ===")
print("Format: JSON with yaw, roll, joy1_x, joy1_y, fire")
print("Connect the PC visualiser now!\n")

FIRE_THRESHOLD = 25  # joystick deflection to count as "firing"

loop_count = 0
last_print = ticks_ms()

while True:
    # Update IMU (integrates yaw, filters roll)
    yaw, roll, gz = imu.update()
    
    # Read joystick 1: tank movement
    steer = read_joystick(joy1_x, joy_center['j1x'])    # -100 (left) to +100 (right)
    throttle = read_joystick(joy1_y, joy_center['j1y'])  # -100 (back) to +100 (forward)
    
    # Read joystick 2: fire trigger (any Y deflection = fire)
    fire_raw = read_joystick(joy2_y, joy_center['j2y'])
    fire = 1 if abs(fire_raw) > FIRE_THRESHOLD else 0
    
    # Build data packet
    packet = {
        'yaw': round(yaw, 1),
        'roll': round(roll, 1),
        'steer': steer,
        'throttle': throttle,
        'fire': fire,
    }
    
    # Send as JSON over USB serial (PC reads this)
    print(json.dumps(packet))
    
    # Sleep to maintain ~50Hz
    sleep_ms(20)
