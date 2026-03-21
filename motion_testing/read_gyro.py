# read_gyro.py — Quick gyro diagnostic tool
# Connects to the Pico and prints live sensor data
# Usage: python read_gyro.py

import sys
import json
import serial
import serial.tools.list_ports


def find_pico_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or '').lower()
        if 'pico' in desc or 'board' in desc or 'usb serial' in desc or 'usbmodem' in desc:
            return p.device
    if ports:
        return ports[0].device
    return None


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_pico_port()
    if not port:
        print("ERROR: No serial port found. Is the Pico plugged in?")
        sys.exit(1)

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 115200, timeout=1)
    ser.reset_input_buffer()
    print("Reading gyro data (Ctrl+C to stop):\n")

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line.startswith('{'):
                continue
            try:
                d = json.loads(line)
                print(
                    f"\ryaw={d['yaw']:>7.1f}  roll={d['roll']:>7.1f}  "
                    f"steer={d['steer']:>4d}  throttle={d['throttle']:>4d}  "
                    f"fire={'YES' if d['fire'] else ' - '}",
                    end='', flush=True,
                )
            except (json.JSONDecodeError, KeyError):
                pass
    except KeyboardInterrupt:
        print("\n\nDone.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
