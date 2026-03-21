# ROBO-HUNTER — Hackabot 2026

A physical laser tag game. Aim a real gun at a moving robot and shoot — the laptop shows the robot's live camera feed as your FPS game view.

---

## Quick Start

```
conda activate qiskit-env
cd "Hackabot-2026"
pip install opencv-python pygame numpy ultralytics scipy pyserial
python "video streaming.py"
```

---

## Phone Stream Setup (Android)

1. Install **IP Webcam** (Play Store, free)
2. Open app → scroll to bottom → **Start server**
3. Note the IP (e.g. `10.0.0.5`)
4. In `video streaming.py`:
   ```python
   PHONE_IP = "10.0.0.5"
   USE_WEBCAM_FALLBACK = False
   ```
5. Phone and laptop must be on the same WiFi

**iOS:** Use **DroidCam** (port 4747) or **IP Camera Lite** (port 8080) — same steps.

**No phone?** Set `USE_WEBCAM_FALLBACK = True` — uses the laptop webcam.

---

## Controls

| Key / Action  | Effect                        |
|---------------|-------------------------------|
| Mouse move    | Aim crosshair (testing only)  |
| Left click / Space | Shoot                   |
| R             | Reload                        |
| D             | Simulate taking damage        |
| [ / ]         | Volume down / up              |
| F11           | Toggle fullscreen             |
| ESC           | Quit                          |

When **Pico 1** is connected via USB, mouse aim is replaced by the IMU sensor.

---

## Hardware Setup

### Components

| Component       | Role                                                    |
|-----------------|---------------------------------------------------------|
| Pico 1 (gun)    | BMI160 IMU aiming, joystick fire trigger, reload button, IR LED, USB serial to laptop |
| Pico 2 (robot)  | Autonomous movement, servo launcher, IR receiver        |
| nRF24L01+ ×2    | Wireless link — Pico 1 ↔ Pico 2                        |
| L298N motor driver | Controls robot DC motors                             |
| TSOP38238       | 38kHz IR receiver on robot — detects hits               |
| Phone (on robot)| Streams camera to laptop over WiFi                      |

---

### Pico 1 Wiring (Gun)

| Component        | Pico pin                      |
|------------------|-------------------------------|
| BMI160 SDA       | GP4                           |
| BMI160 SCL       | GP5                           |
| nRF24 SCK        | GP10                          |
| nRF24 MOSI       | GP11                          |
| nRF24 MISO       | GP12                          |
| nRF24 CSN        | GP13                          |
| nRF24 CE         | GP9                           |
| Joystick 1 X (steer) | GP26                     |
| Joystick 1 Y (throttle) | GP27                  |
| Joystick 2 Y (fire trigger) | GP28              |
| Reload button    | GP15 → GND (pull-up)          |
| IR LED           | GP16 → 100Ω → GND             |

### Pico 2 Wiring (Robot)

| Component        | Pico pin                      |
|------------------|-------------------------------|
| nRF24 SCK        | GP2                           |
| nRF24 MOSI       | GP3                           |
| nRF24 MISO       | GP4                           |
| nRF24 CSN        | GP5                           |
| nRF24 CE         | GP6                           |
| IR Receiver OUT  | GP16                          |
| L298N ENA (PWM)  | GP10                          |
| L298N IN1        | GP11                          |
| L298N IN2        | GP12                          |
| L298N ENB (PWM)  | GP13                          |
| L298N IN3        | GP14                          |
| L298N IN4        | GP15                          |
| Head pan servo   | GP20 (yaw — left/right)       |
| Head tilt servo  | GP21 (pitch — up/down)        |
| Launcher servo   | GP22                          |

---

### Flashing the Picos

1. Install **Thonny** or use `mpremote`
2. Flash **MicroPython** firmware onto each Pico (download from micropython.org)
3. Copy `pico1_gun.py` + `bmi160.py` → Pico 1 and save `pico1_gun.py` as `main.py`
4. Copy `pico2_robot.py` → Pico 2 and save as `main.py`

---

### Enabling Serial on the Laptop

1. Plug Pico 1 into the laptop via USB
2. Open Device Manager → Ports (COM & LPT) → note the COM port (e.g. `COM4`)
3. In `video streaming.py`:
   ```python
   ENABLE_SERIAL = True
   SERIAL_PORT   = "COM4"   # your port here
   ```

---

## Serial Protocol (Pico 1 → Laptop)

All messages are newline-terminated ASCII over USB serial at 115200 baud.

| Message         | Meaning                              |
|-----------------|--------------------------------------|
| `AIM:yaw,roll`  | IMU angles → moves crosshair        |
| `SHOOT`         | Fire trigger deflected → fire        |
| `RELOAD`        | Reload button pressed                |
| `HIT`           | Robot's IR receiver was triggered    |

**Re-zero the aim:** Hold reload + trigger simultaneously to reset yaw/pitch to centre.

---

## Tuning

All tuning values are at the top of each file.

**`video streaming.py`**
```python
AIM_YAW_RANGE   = 45.0   # degrees of physical gun sweep = full screen width
AIM_PITCH_RANGE = 20.0   # degrees of physical gun sweep = full screen height
AIM_SMOOTHING   = 0.25   # crosshair smoothing (0 = raw, 1 = frozen)
SOUND_VOLUME    = 0.8    # gunshot/hit/damage volume
RELOAD_VOLUME   = 1.0    # reload sound volume
MUSIC_VOLUME    = 0.4    # background music volume
FULLSCREEN      = True   # launch fullscreen; F11 toggles
```

**`pico1_gun.py`**
```python
SEND_HZ         = 50     # AIM update rate (Hz)
FIRE_THRESHOLD  = 25     # joystick deflection % to count as firing
JOY_DEADZONE    = 4000   # ADC units — ignores small joystick drift
```

---

## Project Structure

```
Hackabot-2026/
├── video streaming.py   # Main game — laptop FPS display, detection, sounds
├── pico1_gun.py         # Pico 1 firmware — gun controller (IMU + joystick + nRF24)
├── pico2_robot.py       # Pico 2 firmware — robot controller (motors + servos + IR)
├── bmi160.py            # BMI160 IMU driver (used by pico1_gun.py)
├── yolov8n.pt           # YOLOv8 nano model for human detection
├── revolver_shot.wav    # Gunshot sound effect
├── revolver_reload.mp3  # Reload sound effect
├── game_start_music.mp3 # In-game background music
├── menu_music.mp3       # Menu background music
└── motion_testing/      # IMU testing & visualisation utilities
    ├── main.py
    ├── read_gyro.py
    ├── visualiser.py
    └── bmi160.py
```
