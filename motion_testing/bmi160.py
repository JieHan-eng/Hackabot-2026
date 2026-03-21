# bmi160.py — BMI160 driver for Pico 2 with yaw integration + complementary filter
# For the gun controller: outputs roll (tilt up/down) and yaw (rotate left/right)

from machine import I2C
from time import sleep_ms, ticks_ms, ticks_diff
import struct
import math

BMI160_ADDR = 0x68

REG_CHIP_ID   = 0x00
REG_DATA_GYR  = 0x0C
REG_DATA_ACC  = 0x12
REG_CMD       = 0x7E
REG_ACC_CONF  = 0x40
REG_ACC_RANGE = 0x41
REG_GYR_CONF  = 0x42
REG_GYR_RANGE = 0x43

CMD_SOFT_RESET = 0xB6
CMD_ACC_NORMAL = 0x11
CMD_GYR_NORMAL = 0x15


class BMI160:
    def __init__(self, i2c, addr=BMI160_ADDR):
        self.i2c = i2c
        self.addr = addr
        self._acc_scale = 16384.0   # +/-2g
        self._gyr_scale = 131.2     # +/-250 dps

        # Calibration offsets
        self.gyr_offset = [0.0, 0.0, 0.0]

        # Integrated yaw angle (degrees)
        self.yaw = 0.0
        self.roll = 0.0
        self._last_time = None

    def _read_reg(self, reg, n=1):
        return self.i2c.readfrom_mem(self.addr, reg, n)

    def _write_reg(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val]))

    def init(self):
        """Initialize BMI160 sensor."""
        self._write_reg(REG_CMD, CMD_SOFT_RESET)
        sleep_ms(100)

        chip_id = self._read_reg(REG_CHIP_ID)[0]
        if chip_id != 0xD1:
            raise RuntimeError(f"BMI160 not found! Got chip ID: {hex(chip_id)}")

        self._write_reg(REG_CMD, CMD_ACC_NORMAL)
        sleep_ms(50)
        self._write_reg(REG_CMD, CMD_GYR_NORMAL)
        sleep_ms(100)

        # 100Hz ODR, normal filter, +/-2g accel, +/-250dps gyro
        self._write_reg(REG_ACC_CONF, 0x28)
        self._write_reg(REG_ACC_RANGE, 0x03)
        self._write_reg(REG_GYR_CONF, 0x28)
        self._write_reg(REG_GYR_RANGE, 0x03)
        sleep_ms(50)

        self._last_time = ticks_ms()
        print("BMI160 OK (chip 0xD1)")

    def calibrate(self, samples=200, delay_ms=10):
        """Calibrate gyro offsets. Keep sensor FLAT and STILL!"""
        print(f"Calibrating ({samples} samples)... keep still!")
        sleep_ms(1000)

        sx, sy, sz = 0.0, 0.0, 0.0
        for _ in range(samples):
            data = self._read_reg(REG_DATA_GYR, 6)
            gx, gy, gz = struct.unpack('<hhh', data)
            sx += gx / self._gyr_scale
            sy += gy / self._gyr_scale
            sz += gz / self._gyr_scale
            sleep_ms(delay_ms)

        self.gyr_offset = [sx / samples, sy / samples, sz / samples]
        print(f"Offsets: x={self.gyr_offset[0]:.2f} y={self.gyr_offset[1]:.2f} z={self.gyr_offset[2]:.2f} dps")

        # Reset orientation
        self.yaw = 0.0
        self.roll = 0.0
        self._last_time = ticks_ms()
        print("Calibration done! Yaw and roll zeroed.")

    def update(self):
        """Read sensor and update yaw + roll angles. Call this every loop iteration.
        
        Roll = from accelerometer (tilt up/down) -- stable, no drift
        Yaw = integrated from gyroscope Z (rotate left/right) -- will drift slowly
        
        Returns: (yaw_deg, roll_deg, raw_gyro_z_dps)
        """
        now = ticks_ms()
        dt = ticks_diff(now, self._last_time) / 1000.0  # seconds
        self._last_time = now

        # Clamp dt to avoid huge jumps
        if dt > 0.1:
            dt = 0.02

        # Read all 12 bytes: gyro xyz + accel xyz
        data = self._read_reg(REG_DATA_GYR, 12)
        gx_r, gy_r, gz_r, ax_r, ay_r, az_r = struct.unpack('<hhhhhh', data)

        # Convert to physical units
        gx = gx_r / self._gyr_scale - self.gyr_offset[0]
        gy = gy_r / self._gyr_scale - self.gyr_offset[1]
        gz = gz_r / self._gyr_scale - self.gyr_offset[2]
        ax = ax_r / self._acc_scale
        ay = ay_r / self._acc_scale
        az = az_r / self._acc_scale

        # Dead zone: ignore tiny gyro noise
        if abs(gz) < 0.5:
            gz = 0.0

        # Yaw: integrate gyroscope Z axis
        # Negate so rotating gun RIGHT = positive yaw
        self.yaw += -gz * dt

        # Keep yaw in -180 to 180
        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw < -180:
            self.yaw += 360

        # Roll from accelerometer (tilt gun barrel up/down)
        if az == 0:
            az = 0.001
        accel_roll = math.atan2(-ax, az) * 180.0 / math.pi

        # Complementary filter: blend accel (stable) with gyro (smooth)
        alpha = 0.96
        self.roll = alpha * (self.roll + gx * dt) + (1.0 - alpha) * accel_roll

        return self.yaw, self.roll, gz

    def reset_yaw(self):
        """Reset yaw to 0 (re-center)."""
        self.yaw = 0.0
