# read_gyro.py — Gun diagnostic tool
# Connects to the Pico and prints live AIM/SHOOT/RELOAD/HIT messages
# Usage: python read_gyro.py  (auto-detects port)
#        python read_gyro.py COM6  (specify port manually)

import sys
import serial
import serial.tools.list_ports


def find_pico_port():
    ports = serial.tools.list_ports.comports()
    # Prefer USB Serial, skip Bluetooth
    for p in ports:
        desc = (p.description or '').lower()
        if 'usb serial' in desc and 'bluetooth' not in desc:
            return p.device
    for p in ports:
        desc = (p.description or '').lower()
        if 'bluetooth' not in desc:
            return p.device
    return None


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_pico_port()
    if not port:
        print("ERROR: No serial port found. Is the Pico plugged in?")
        sys.exit(1)

    print(f"Connecting to {port} at 115200 baud...")
    ser = serial.Serial(port, 115200, timeout=1)
    ser.reset_input_buffer()
    print("Waiting for Pico to start (calibration takes ~5 seconds)...\n")
    print(f"{'YAW':>8}  {'ROLL':>8}  EVENTS")
    print("-" * 40)

    yaw = roll = 0.0
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            if line.startswith("AIM:"):
                try:
                    parts = line[4:].split(",")
                    yaw  = float(parts[0])
                    roll = float(parts[1])
                    print(f"\r{yaw:>8.1f}  {roll:>8.1f}  ", end='', flush=True)
                except (ValueError, IndexError):
                    pass

            elif line in ("SHOOT", "RELOAD", "HIT"):
                print(f"\r{yaw:>8.1f}  {roll:>8.1f}  *** {line} ***")

            else:
                # Print startup/calibration messages from the Pico as-is
                print(f"  [{line}]")

    except KeyboardInterrupt:
        print("\n\nDone.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
