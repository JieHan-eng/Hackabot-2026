# ROBO-HUNTER — Hackabot 2026

A physical laser tag game. Aim a real gun at a moving robot and shoot — the laptop shows the robot's live camera feed as your FPS game view.

---

## Quick Start

```
conda activate qiskit-env
cd "Hackabot-2026"
pip install opencv-python pygame numpy ultralytics scipy
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
| Pico 1 (gun)    | BMI160 IMU aiming, trigger, reload button, IR LED, USB serial to laptop |
| Pico 2 (robot)  | Autonomous movement, servo launcher, IR receiver        |
| nRF24L01+ ×2   | Wireless link — Pico 1 ↔ Pico 2                        |
| L298N motor driver | Controls robot DC motors                            |
| TSOP38238       | 38kHz IR receiver on robot — detects hits               |
| Phone (on robot)| Streams camera to laptop over WiFi                     |

---

### Pico 1 Wiring (Gun)

| Component        | Pico pin                      |
|------------------|-------------------------------|
| BMI160 SDA       | GP0                           |
| BMI160 SCL       | GP1                           |
| nRF24 SCK        | GP2                           |
| nRF24 MOSI       | GP3                           |
| nRF24 MISO       | GP4                           |
| nRF24 CSN        | GP5                           |
| nRF24 CE         | GP6                           |
| Joystick X (ADC) | GP26                          |
| Joystick Y (ADC) | GP27                          |
| Joystick button  | GP28 → GND (pull-up)          |
| Trigger button   | GP14 → GND (pull-up)          |
| Reload button    | GP15 → GND (pull-up)          |
| IR LED           | GP16 → 100Ω → GND            |

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
3. Copy `pico1_gun.py` → Pico 1 and save as `main.py`
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
| `AIM:yaw,pitch` | IMU angles → moves crosshair         |
| `SHOOT`         | Trigger pulled → fire                |
| `RELOAD`        | Reload button pressed                |
| `HIT`           | Robot's IR receiver was triggered    |

**Re-zero the aim:** Hold reload + trigger simultaneously to reset yaw/pitch to centre.

---

## Tuning

All tuning values are at the top of each file.

**`video streaming.py`**
```python
AIM_YAW_RANGE   = 45.0   # degrees of physical gun sweep = full screen width
AIM_PITCH_RANGE = 30.0   # degrees of physical gun sweep = full screen height
AIM_SMOOTHING   = 0.25   # crosshair smoothing (0 = raw, 1 = frozen)
SOUND_VOLUME    = 0.8    # gunshot/hit/damage volume
RELOAD_VOLUME   = 1.0    # reload sound volume
FULLSCREEN      = True   # launch fullscreen; F11 toggles
```

**`pico1_gun.py`**
```python
YAW_RANGE       = 45.0   # must match AIM_YAW_RANGE above
PITCH_RANGE     = 30.0   # must match AIM_PITCH_RANGE above
YAW_DEADZONE    = 0.8    # deg/s below which gyro drift is ignored
SEND_HZ         = 50     # AIM update rate
```
