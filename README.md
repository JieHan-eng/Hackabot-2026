# Gun Controller — Tank FPV Project

## Project structure

```
gun-controller/
├── pico/               # Upload these to the Raspberry Pi Pico 2
│   ├── bmi160.py       # BMI160 IMU driver (yaw + roll)
│   └── main.py         # Main firmware (reads IMU + joysticks, sends JSON over serial)
│
├── pc/                 # Run these on your computer
│   └── visualiser.py   # 3D visualiser (pygame + OpenGL)
│
└── README.md
```

## Wiring

### BMI160 (IMU)
| BMI160 pin | Pico pin     | Physical pin |
|------------|-------------|-------------|
| SDA        | GP4         | Pin 6       |
| SCL        | GP5         | Pin 7       |
| VCC        | 3V3(OUT)    | Pin 36      |
| GND        | GND         | Pin 38      |

### Joystick 1 (tank movement)
| Joystick pin | Pico pin | Physical pin |
|-------------|---------|-------------|
| VRx         | GP26    | Pin 31      |
| VRy         | GP27    | Pin 32      |
| VCC         | 3V3     | Pin 36      |
| GND         | GND     | Pin 38      |

### Joystick 2 (fire trigger)
| Joystick pin | Pico pin | Physical pin |
|-------------|---------|-------------|
| VRy         | GP28    | Pin 34      |
| VCC         | 3V3     | Pin 36      |
| GND         | GND     | Pin 38      |

## How to upload code to the Pico 2

### Option A: Thonny (easiest)

1. Open Thonny IDE
2. Go to Tools > Options > Interpreter > select "MicroPython (Raspberry Pi Pico)"
3. Open `pico/bmi160.py` in Thonny
4. File > Save As > choose "Raspberry Pi Pico" > save as `bmi160.py`
5. Open `pico/main.py` in Thonny
6. File > Save As > choose "Raspberry Pi Pico" > save as `main.py`
7. Press the green Run button (or F5) to start

### Option B: mpremote (command line, works great with VS Code / Claude Code)

Install mpremote:
```bash
pip install mpremote
```

Upload files:
```bash
cd gun-controller

# Copy both files to the Pico
mpremote cp pico/bmi160.py :bmi160.py
mpremote cp pico/main.py :main.py

# Soft reboot to start running main.py
mpremote reset

# To see live serial output
mpremote connect

# To run a file without saving it permanently
mpremote run pico/main.py
```

### Option C: VS Code with MicroPico extension

1. Install the "MicroPico" extension in VS Code (search for "micropico")
2. Open this folder in VS Code
3. Click the MicroPico status bar icon to connect to the Pico
4. Right-click `pico/bmi160.py` > "Upload file to Pico"
5. Right-click `pico/main.py` > "Upload file to Pico"
6. Use the MicroPico terminal to see output

## Running the 3D visualiser

```bash
# Install dependencies (once)
pip install pyserial pygame PyOpenGL

# Make sure Thonny is closed (it holds the serial port)
# Run the visualiser
python pc/visualiser.py

# Or specify port manually
python pc/visualiser.py COM3          # Windows
python pc/visualiser.py /dev/ttyACM0  # Linux
python pc/visualiser.py /dev/tty.usbmodem1234  # Mac
```

## Data packet format

The Pico sends JSON at ~50Hz over USB serial:
```json
{"yaw": 15.3, "roll": -8.2, "steer": 45, "throttle": 80, "fire": 0}
```

| Field    | Range        | Meaning                          |
|----------|-------------|----------------------------------|
| yaw      | -180 to 180 | Gun rotation left/right (degrees)|
| roll     | -90 to 90   | Gun tilt up/down (degrees)       |
| steer    | -100 to 100 | Joystick 1 X axis (tank steer)   |
| throttle | -100 to 100 | Joystick 1 Y axis (tank drive)   |
| fire     | 0 or 1      | Joystick 2 pulled = firing       |
