# ROBO-HUNTER — Hackabot 2026

A physical laser tag game with a live FPS display. Aim your gun at a moving robot and shoot it in real life — the laptop shows the robot's camera as your game view.

---

## Requirements

```
pip install opencv-python pygame numpy ultralytics
```

> Run inside the **qiskit-env** conda environment:
> `conda activate qiskit-env`

---

## Run

```
cd "Hackabot-2026"
python "video streaming.py"
```

---

## Phone stream setup (Android)

1. Install **IP Webcam** (Play Store, free)
2. Open app → scroll to bottom → **Start server**
3. Note the IP shown (e.g. `10.0.0.5`)
4. In `video streaming.py`, set:
   ```python
   PHONE_IP = "10.0.0.5"
   USE_WEBCAM_FALLBACK = False
   ```
5. Phone and laptop must be on the **same WiFi** (use phone hotspot if on campus)

**iOS:** Use **DroidCam** (port 4747) or **IP Camera Lite** (port 8080) — same steps.

---

## Webcam mode (no phone)

Set `USE_WEBCAM_FALLBACK = True` in `video streaming.py` — uses your laptop camera.

---

## Controls

| Key / Action | Effect |
|---|---|
| Mouse move | Aim crosshair |
| Left click / Space | Shoot |
| R | Reload |
| D | Simulate taking damage |
| ESC | Quit |

When the **Pico gun** is connected via USB, mouse aim is replaced by the IMU sensor.

---

## Hardware (full setup)

| Component | Role |
|---|---|
| Pico 1 (gun) | IMU aiming, IR shoot, USB serial to laptop |
| Pico 2 (robot) | Autonomous movement, servo launcher, IR receiver |
| nRF24L01+ modules | Wireless between Pico 1 and Pico 2 |
| Phone on robot | Streams camera to laptop over WiFi |

**Pico serial config** — edit in `video streaming.py`:
```python
ENABLE_SERIAL = True
SERIAL_PORT   = "COM3"   # check Device Manager for your port
```

Pico sends over USB serial (newline-terminated):
- `AIM:yaw,pitch` — moves crosshair
- `SHOOT` — fires
- `HIT` — player takes damage
- `JOY:x,y` — joystick
