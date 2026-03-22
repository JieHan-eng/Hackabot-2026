# ROBO-HUNTER — Hackabot 2026

A physical FPS game. Aim a real gun controller at a robot tank, shoot targets detected on its live camera feed, and drive the tank with a joystick — all displayed as an FPS view on your laptop.

---

## Quick Start

```
conda activate qiskit-env
cd "Hackabot-2026"
pip install opencv-python pygame numpy ultralytics scipy pyserial
python "video streaming.py"
```

---

## Phone Stream Setup

The phone is strapped to the robot tank for a POV camera — it is not wired to the tank.

1. Install **IP Webcam** (Android, free) or **DroidCam** (iOS)
2. Open app → Start server → note the IP shown (e.g. `10.0.0.5`)
3. In `video streaming.py`, set:
   ```python
   PHONE_IP = "10.0.0.5"
   ```
4. Phone and laptop must be on the **same WiFi**

**No phone?** Set `USE_WEBCAM_FALLBACK = True` to use the laptop webcam instead.

---

## Controls

### Gun Controller (Pico 1)

| Action | Effect |
|--------|--------|
| Move gun physically | Aim crosshair (IMU tracking) |
| Joystick 1 (left stick) | Drive tank (forward/back/steer) |
| Fire joystick — pull | Shoot |
| Fire joystick — push | Centre crosshair |
| Fire joystick — press down (SW) | Centre crosshair (backup) |
| Flick gun up-and-down | Reload |

### Keyboard

| Key | Effect |
|-----|--------|
| C | Centre crosshair |
| Space | Shoot (testing without gun) |
| R | Reload |
| D | Simulate taking damage |
| [ / ] | Volume down / up |
| F11 | Toggle fullscreen |
| ESC | Quit / back to menu |

---

## Hardware

### Components

| Component | Role |
|-----------|------|
| Pico 1 (gun) | BMI160 IMU aiming, 2 joysticks, nRF24 radio, USB serial to laptop |
| Pico 2 (tank) | Receives commands via nRF24, drives DC motors |
| nRF24L01+ x2 | Wireless link between gun and tank |
| L298N | Motor driver for tank DC motors |
| Phone | Strapped to tank, streams camera to laptop over WiFi |

### Pico 1 Wiring (Gun)

| Component | Pin |
|-----------|-----|
| BMI160 SDA / SCL | GP4 / GP5 |
| Joystick 1 X (steer) | GP26 |
| Joystick 1 Y (throttle) | GP27 |
| Joystick 2 Y (fire trigger) | GP28 |
| Joystick 2 SW (centre aim) | GP22 |
| nRF24 SCK / MOSI / MISO | GP18 / GP19 / GP16 |
| nRF24 CE / CSN | GP17 / GP20 |
| IR LED (future) | GP14 |
| Reload button (legacy) | GP15 |

### Pico 2 Wiring (Tank)

| Component | Pin |
|-----------|-----|
| nRF24 SCK / MOSI / MISO | GP18 / GP19 / GP16 |
| nRF24 CE / CSN | GP17 / GP20 |
| L298N IN1 / IN2 / ENA | GP2 / GP3 / GP6 (left motor) |
| L298N IN3 / IN4 / ENB | GP7 / GP8 / GP9 (right motor) |

---

## Flashing the Picos

1. Hold BOOTSEL and plug in the Pico via USB
2. Copy the MicroPython `.uf2` file to the drive that appears
3. Wait for reboot, then upload files with `mpremote`:

**Gun Pico:**
```
mpremote cp pico1_gun.py :main.py
mpremote cp bmi160.py :bmi160.py
mpremote cp nrf24l01.py :nrf24l01.py
```

**Tank Pico:**
```
mpremote cp pico2_robot.py :main.py
mpremote cp nrf24l01.py :nrf24l01.py
```

---

## Serial Protocol (Gun → Laptop)

USB serial at 115200 baud. Messages are newline-terminated.

| Message | Meaning |
|---------|---------|
| `AIM:yaw,roll` | IMU angles — moves crosshair |
| `SHOOT` | Fire trigger pulled |
| `RELOAD` | Gun flicked up-and-down (gyro gesture) |
| `ZERO` | Joystick pushed or SW pressed — centres crosshair |

---

## Gameplay

- Each round lasts **30 seconds**
- Kill detected targets to score — **1 kill = 1 point**
- Timer displays at the top of the screen (flashes red when <10s)
- When time runs out, if you're in the **top 5**, you enter your name for the leaderboard

## Leaderboard

- Accessible from the map select menu (scroll down past the maps)
- Displays the **top 5 scores** with player name, kills, and map played
- Data is saved to `leaderboard.json` and **persists between sessions**

---

## Maps

5 selectable maps, each with unique visual filters and atmosphere:

- **Arctic Storm** — blizzard, frost edges, snow particles, breath fog
- **Warzone** — sepia grade, rubble silhouette, smoke, firelight flicker
- **Jungle Warfare** — green tint, vine borders, rain, lightning
- **Night Ops** — night-vision goggles, NV noise, scope sway, phosphor glow
- **Cyberpunk City** — neon edges, chromatic aberration, glitch effects, holo grid

Detection (YOLOv8) runs on the **raw camera feed** before filters are applied.

---

## Project Structure

```
Hackabot-2026/
├── video streaming.py    # Main game (laptop)
├── pico1_gun.py          # Gun Pico firmware
├── pico2_robot.py        # Tank Pico firmware
├── bmi160.py             # BMI160 IMU driver
├── nrf24l01.py           # nRF24L01+ radio driver
├── yolov8n.pt            # YOLOv8 nano model
├── revolver_shot.wav     # Gunshot sound
├── revolver_reload.mp3   # Reload sound
├── menu_select.mp3       # Menu selection sound
├── game_start_music.mp3  # In-game music
├── menu_music.mp3        # Menu music
└── leaderboard.json      # Persistent top 5 scores (auto-created)
```
