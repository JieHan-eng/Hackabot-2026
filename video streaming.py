import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
import threading
import queue

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[!] ultralytics not installed — falling back to HOG detector")
    print("    To enable YOLO:  pip install ultralytics")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[!] pyserial not installed — physical gun input disabled")
    print("    To enable serial:  pip install pyserial")

# ═══════════════════════════════════════════════════════════════
#  PHONE STREAM CONFIG
#
#  ── Quick setup ─────────────────────────────────────────────
#  1. Connect your phone and laptop to the same WiFi network.
#  2. Install a streaming app on your phone:
#
#     ANDROID  →  "IP Webcam" by Pavel Khlebovich  (Play Store, free)
#                 Open the app → scroll to bottom → "Start server"
#                 Note the IP shown, e.g.  192.168.1.42
#
#     iOS      →  "DroidCam" by Dev47Apps  (App Store, free)
#                 OR  "IP Camera Lite"     (App Store, free)
#                 Open app → tap Start → note the IP shown
#
#  3. Set PHONE_IP below to that IP address (numbers only, no http://).
#  4. Set USE_WEBCAM_FALLBACK = False.
#  5. Run the script — it will automatically try every known URL
#     for both Android and iOS until one works.
#
#  ── Testing without a phone ─────────────────────────────────
#  Set USE_WEBCAM_FALLBACK = True to use your laptop's webcam.
#
#  ── URL reference (for manual testing in a browser) ─────────
#  Android  IP Webcam :  http://IP:8080/video
#                        http://IP:8080/videofeed   (MJPEG alt)
#  iOS  DroidCam      :  http://IP:4747/video
#  iOS  IP Camera Lite:  http://IP:8080/live        (MJPEG)
#                        rtsp://IP:554/live          (lower latency)
# ═══════════════════════════════════════════════════════════════

PHONE_IP = "YOUR_PHONE_IP"  # ← SET THIS to your phone's IP address

# True  = use laptop webcam (good for testing)
# False = connect to phone stream (set PHONE_IP first)
USE_WEBCAM_FALLBACK = True

# All URLs tried in order when USE_WEBCAM_FALLBACK = False.
# The script auto-detects which one works — no need to pick manually.
STREAM_CANDIDATES = [
    # ── Android: IP Webcam ──────────────────────────────────
    f"http://{PHONE_IP}:8080/video",       # primary stream
    f"http://{PHONE_IP}:8080/videofeed",   # MJPEG alternate
    # ── iOS: DroidCam ───────────────────────────────────────
    f"http://{PHONE_IP}:4747/video",
    # ── iOS: IP Camera Lite ─────────────────────────────────
    f"http://{PHONE_IP}:8080/live",        # MJPEG
    f"rtsp://{PHONE_IP}:554/live",         # RTSP (lower latency)
    # ── Generic fallbacks (some apps share these ports) ─────
    f"http://{PHONE_IP}:8080/mjpeg",
    f"http://{PHONE_IP}:4747/mjpeg",
    f"http://{PHONE_IP}:8080/stream",
]

# How long (seconds) to wait per URL before trying the next one
STREAM_CONNECT_TIMEOUT = 4

WINDOW_W = 1280
WINDOW_H = 720

# ─── DETECTION CONFIG ─────────────────────────────────────────
#
#  ENABLE_DETECTION : toggle human detection on/off
#  USE_YOLO         : True  = YOLOv8n (pip install ultralytics, downloads ~6 MB model)
#                     False = OpenCV HOG (no extra install, less accurate)
#  YOLO_MODEL_PATH  : model filename — "yolov8n.pt" auto-downloads on first run
#  YOLO_CONF        : detection confidence threshold (0–1)
#
ENABLE_DETECTION = True
USE_YOLO         = True
YOLO_MODEL_PATH  = "yolov8n.pt"
YOLO_CONF        = 0.45

# HIT_SHRINK_X / HIT_SHRINK_Y : how much to shrink the YOLO box for hit detection
#   X shrinks left+right edges inward (makes hitbox narrow like a person's body)
#   Y shrinks top+bottom edges inward (small value keeps head-to-toe coverage)
#   0.25 X means each side moves in 25% → final width is 50% of the original box
HIT_SHRINK_X      = 0.25
HIT_SHRINK_Y      = 0.05

# BOX_SMOOTH_SPEED: how fast the displayed box glides to the new detected position (per second)
#                   Higher = snappier, lower = smoother. 8 is a good balance.
BOX_SMOOTH_SPEED  = 4.0

# ─── SOUND CONFIG ─────────────────────────────────────────────
#
#  SOUND_VOLUME : master volume for all sound effects (0.0 = silent, 1.0 = full)
#
SOUND_VOLUME = 0.8

# ─── SERIAL (Pico 1 USB) CONFIG ───────────────────────────────
#
#  1. Plug Pico 1 (gun) into your laptop via USB
#  2. Check Device Manager → Ports (COM & LPT) for the COM port number
#  3. Set SERIAL_PORT to that port (e.g. "COM4")
#  4. Set ENABLE_SERIAL = True
#
#  Protocol — Pico 1 sends these lines (newline-terminated):
#
#    AIM:yaw,pitch  — absolute IMU angles in degrees, e.g. AIM:12.4,-5.1
#                     yaw  = left/right gun rotation  → crosshair X
#                     pitch= up/down gun rotation     → crosshair Y
#
#    SHOOT          — trigger pulled (fires IR LED toward robot)
#    HIT            — gun's IR receiver was hit (player takes damage)
#    RELOAD         — reload button pressed on gun handle
#    JOY:x,y        — joystick values  -100 to +100, e.g. JOY:0,75
#                     forwarded to robot via nRF24 for movement
#
ENABLE_SERIAL = False
SERIAL_PORT   = "COM3"    # ← CHANGE THIS to your Pico's COM port
SERIAL_BAUD   = 115200

# ─── AIM / CROSSHAIR CONFIG ───────────────────────────────────
#
#  AIM_YAW_RANGE   : gun yaw angle (degrees) that maps to screen edge left/right
#                    e.g. 45 means ±45° of yaw covers the full screen width
#  AIM_PITCH_RANGE : same but for up/down
#  AIM_SMOOTHING   : 0.0 = raw (jittery), 1.0 = never moves. 0.25 is a good start.
#
AIM_YAW_RANGE   = 45.0    # degrees — tune to how wide your physical sweep is
AIM_PITCH_RANGE = 30.0    # degrees
AIM_SMOOTHING   = 0.25    # low-pass smoothing on crosshair position (0 = off)

# ─── COLOURS ──────────────────────────────────
BLACK       = (0,   0,   0)
WHITE       = (255, 255, 255)
RED         = (220, 30,  30)
GREEN       = (0,   255, 120)
DARK_GREEN  = (0,   180, 80)
CYAN        = (0,   230, 255)
ORANGE      = (255, 140, 0)
YELLOW      = (255, 220, 0)
DARK_GREY   = (20,  20,  20)
BLOOD_RED   = (180, 0,   0)
HUD_GREEN   = (0,   255, 140)
HUD_DIM     = (0,   140, 80)
CROSSHAIR_C = (255, 255, 255)


# ─── NPC ──────────────────────────────────────────────────────

class NPC:
    """Tracks a single detected person between frames."""
    MAX_HEALTH = 100

    def __init__(self, x1, y1, x2, y2):
        self.box            = (x1, y1, x2, y2)
        self.display_box    = (float(x1), float(y1), float(x2), float(y2))  # smoothed
        self.health         = self.MAX_HEALTH
        self.display_health = float(self.MAX_HEALTH)   # smoothed health for bar rendering
        self.last_seen      = time.time()
        self.hit_flash      = 0.0   # seconds of hit-flash remaining
        self.kill_time      = None  # set when health → 0


# ─── DETECTION WORKER ─────────────────────────────────────────

class DetectionWorker:
    """Runs person detection in a background thread so the game loop never blocks."""

    def __init__(self, use_yolo: bool):
        self._detections: list = []
        self._lock       = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._detector   = self._init_detector(use_yolo)
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    # ── initialise detector ──
    def _init_detector(self, use_yolo: bool):
        if use_yolo and YOLO_AVAILABLE:
            print("[*] Loading YOLOv8n model (downloads ~6 MB on first run)…")
            model = _YOLO(YOLO_MODEL_PATH)
            # Warm up so the first frame is not slow
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            model(dummy, classes=[0], verbose=False)
            print("[OK] YOLO ready")
            return ("yolo", model)
        else:
            print("[*] Using OpenCV HOG person detector")
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return ("hog", hog)

    # ── public API ──
    def submit(self, bgr_frame: np.ndarray):
        """Non-blocking frame submission — drops frame if worker is busy."""
        try:
            self._queue.put_nowait(bgr_frame)
        except queue.Full:
            pass

    def get_detections(self) -> list:
        """Returns list of (x1, y1, x2, y2) in screen coordinates."""
        with self._lock:
            return list(self._detections)

    # ── background worker ──
    def _run(self):
        while True:
            frame = self._queue.get()   # blocks until a frame arrives
            kind, detector = self._detector
            boxes = []

            try:
                if kind == "yolo":
                    results = detector(frame, classes=[0], conf=YOLO_CONF,
                                       verbose=False, imgsz=640)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            boxes.append((x1, y1, x2, y2))
                else:
                    # HOG — run on a smaller frame for speed, scale boxes back
                    det_w, det_h = 640, 360
                    small = cv2.resize(frame, (det_w, det_h))
                    rects, _ = detector.detectMultiScale(
                        small, winStride=(8, 8), padding=(4, 4), scale=1.05)
                    sx = WINDOW_W / det_w
                    sy = WINDOW_H / det_h
                    for (x, y, w, h) in rects:
                        boxes.append((int(x*sx), int(y*sy),
                                      int((x+w)*sx), int((y+h)*sy)))
            except Exception as e:
                print(f"[!] Detection error: {e}")

            with self._lock:
                self._detections = boxes


# ─── CAMERA CAPTURE ───────────────────────────────────────────

class CameraCapture:
    """
    Background thread that continuously reads from a VideoCapture and keeps
    only the LATEST frame.  This prevents the stale-frame lag that happens
    when OpenCV's internal buffer fills up (especially over WiFi).
    """

    def __init__(self, cap: cv2.VideoCapture):
        # Minimise the internal OpenCV buffer so old frames don't pile up
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap            = cap
        self._frame          = None          # latest BGR frame, or None
        self._lock           = threading.Lock()
        self._alive          = True
        self.last_frame_time = time.time()   # updated each time a real frame arrives
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    # ── public API ──────────────────────────────────────

    def get_latest(self):
        """Return the newest frame and clear it, or None if nothing new yet."""
        with self._lock:
            f, self._frame = self._frame, None
        return f

    def reconnect(self, url: str) -> bool:
        """Try to swap in a fresh VideoCapture for `url`. Returns True on success."""
        if url == "webcam":
            for idx in range(3):
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
                    new_cap = (cv2.VideoCapture(idx, backend)
                               if backend else cv2.VideoCapture(idx))
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if new_cap.isOpened():
                        ret, f = new_cap.read()
                        if ret and f is not None:
                            self._swap(new_cap)
                            return True
                    new_cap.release()
        else:
            new_cap = cv2.VideoCapture(url)
            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            deadline = time.time() + 3
            while time.time() < deadline:
                if new_cap.isOpened():
                    ret, f = new_cap.read()
                    if ret and f is not None:
                        self._swap(new_cap)
                        return True
                time.sleep(0.15)
            new_cap.release()
        return False

    def release(self):
        self._alive = False
        with self._lock:
            self._cap.release()

    # ── internal ────────────────────────────────────────

    def _swap(self, new_cap: cv2.VideoCapture):
        """Atomically replace the VideoCapture; releases the old one after a delay."""
        with self._lock:
            old, self._cap = self._cap, new_cap
        # Release old cap after a short delay so the read thread has time to exit it
        threading.Thread(
            target=lambda: (time.sleep(0.5), old.release()), daemon=True).start()

    def _run(self):
        while self._alive:
            with self._lock:
                cap = self._cap
            try:
                ret, frame = cap.read()
            except Exception:
                time.sleep(0.01)
                continue
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame   # always overwrite — keep only latest
                self.last_frame_time = time.time()
            else:
                time.sleep(0.005)


# ─── HIT MARKER ───────────────────────────────────────────────

class HitMarker:
    def __init__(self, x, y, confirmed=False):
        self.x         = x
        self.y         = y
        self.confirmed = confirmed   # True = NPC was hit
        self.birth     = time.time()
        self.duration  = 0.5 if confirmed else 0.35

    def alive(self):
        return time.time() - self.birth < self.duration

    def alpha(self):
        return 1.0 - (time.time() - self.birth) / self.duration


# ─── MAIN GAME CLASS ──────────────────────────────────────────

class FPSGame:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.display.set_caption("ROBO-HUNTER // HACKABOT 2026")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_big   = pygame.font.SysFont("Courier New", 48, bold=True)
        self.font_med   = pygame.font.SysFont("Courier New", 24, bold=True)
        self.font_small = pygame.font.SysFont("Courier New", 16)
        self.font_hud   = pygame.font.SysFont("Courier New", 32, bold=True)

        # Game state
        self.score          = 0
        self.health         = 100
        self.ammo           = 30
        self.max_ammo       = 30
        self.kills          = 0
        self.hit_markers    = []
        self.damage_flash   = 0.0
        self.shoot_flash    = 0.0
        self.reloading      = False
        self.reload_start   = 0
        self.reload_time    = 2.0
        self.last_shoot     = 0
        self.shoot_cooldown = 0.15
        self.game_active    = True
        self.fps_display    = 0
        self.frame_times    = []
        self.crosshair_spread = 0

        # Moveable crosshair — driven by IMU when serial is connected,
        # otherwise stays at screen centre (fully playable with keyboard/mouse)
        self.ch_x = float(WINDOW_W // 2)
        self.ch_y = float(WINDOW_H // 2)

        # NPC tracking
        self.npcs: list[NPC] = []
        self._active_url = ""

        # Kill feed
        self.kill_feed: list = []   # list of [timestamp, message]

        # ── Procedural sound effects (numpy-generated, no external files) ──
        self._init_sounds()

        # ── Pre-generated visual overlays ──
        self._vignette  = self._make_vignette()
        self._scanlines = self._make_scanlines()

        # Detection worker (runs in background thread)
        self._det_worker = None
        if ENABLE_DETECTION:
            try:
                self._det_worker = DetectionWorker(
                    use_yolo=USE_YOLO and YOLO_AVAILABLE)
            except Exception as e:
                print(f"[!] Detection worker failed to start: {e}")

        # Serial events from Pico 1
        self._serial_events: queue.Queue = queue.Queue()
        self._serial_conn = None
        if ENABLE_SERIAL and SERIAL_AVAILABLE:
            self._init_serial()
        elif ENABLE_SERIAL and not SERIAL_AVAILABLE:
            print("[!] ENABLE_SERIAL=True but pyserial is not installed")

        # Video stream — wrapped in CameraCapture for lag-free latest-frame access
        raw_cap = self._connect_stream()
        self._capture        = CameraCapture(raw_cap)
        self._last_surface   = None   # last successfully rendered pygame surface
        self._reconnect_after = 0.0   # epoch time before which reconnects are suppressed

    # ── procedural sound generation ──────────────────────────

    @staticmethod
    def _make_sound_array(samples: np.ndarray) -> pygame.mixer.Sound:
        """Convert a mono float32 array (-1..1) to a stereo int16 pygame Sound."""
        clipped = np.clip(samples, -1.0, 1.0)
        mono    = (clipped * 32767).astype(np.int16)
        stereo  = np.column_stack([mono, mono])
        return pygame.sndarray.make_sound(stereo)

    @staticmethod
    def _load_wav(path: str, trim_s: float = None) -> pygame.mixer.Sound:
        """Load a WAV file (any bit-depth, any sample rate) as a pygame Sound.
        Trims to trim_s seconds if given. Resamples to 44100 Hz if needed."""
        import wave as _wave
        from math import gcd
        from scipy.signal import resample_poly

        with _wave.open(path) as f:
            sr  = f.getframerate()
            nch = f.getnchannels()
            sw  = f.getsampwidth()
            raw = f.readframes(f.getnframes())

        # Decode to float32 (handles 8, 16, 24, 32-bit)
        data = np.frombuffer(raw, dtype=np.uint8)
        n_samp = len(data) // sw
        arr = np.zeros(n_samp, dtype=np.int32)
        for i in range(sw):
            arr += data[i::sw].astype(np.int32) << (8 * i)
        bits = sw * 8
        arr[arr >= 2 ** (bits - 1)] -= 2 ** bits          # sign-extend
        audio = arr.astype(np.float32) / (2 ** (bits - 1))
        audio = audio.reshape(-1, nch)

        if trim_s is not None:
            audio = audio[:int(sr * trim_s)]

        if sr != 44100:
            g = gcd(44100, sr)
            audio = resample_poly(audio, 44100 // g, sr // g, axis=0)

        if nch == 1:
            audio = np.column_stack([audio, audio])

        s16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.ascontiguousarray(s16))

    def _init_sounds(self):
        SR = 44100  # sample rate

        # ── snd_shoot: real revolver WAV, trimmed to first shot (~0.25 s) ──
        try:
            self.snd_shoot = self._load_wav("revolver_shot.wav", trim_s=0.25)
        except Exception as e:
            print(f"[!] revolver_shot.wav failed ({e}) — using fallback")
            self.snd_shoot = None

        # ── snd_shoot_hit: same gun sound + short metallic ping on top ──
        # The ping plays simultaneously via a separate channel so you hear
        # a clear "hit confirmed" cue on top of the real gunshot.
        try:
            tink_len = int(SR * 0.07)
            t_tk     = np.linspace(0, 1, tink_len)
            tink     = (np.sin(2*np.pi*900  * t_tk) * 0.6 +
                        np.sin(2*np.pi*1400 * t_tk) * 0.35) * np.exp(-t_tk * 30)
            self.snd_hit_tink = self._make_sound_array(tink * 0.85)
        except Exception as e:
            print(f"[!] snd_hit_tink generation failed: {e}")
            self.snd_hit_tink = None

        # ── snd_reload: real revolver reload WAV ──
        try:
            self.snd_reload = self._load_wav("revolver_reload.wav")
        except Exception as e:
            print(f"[!] snd_reload generation failed: {e}")
            self.snd_reload = None

        # ── snd_hit: high metallic ping ──
        try:
            dur     = int(SR * 0.35)
            t       = np.linspace(0, 1, dur)
            freq    = 1800.0
            data    = np.sin(2 * np.pi * freq * t) * np.exp(-t * 9) * 0.7
            # add a slight harmonic
            data   += np.sin(2 * np.pi * freq * 2.5 * t) * np.exp(-t * 14) * 0.3
            self.snd_hit = self._make_sound_array(data)
        except Exception as e:
            print(f"[!] snd_hit generation failed: {e}")
            self.snd_hit = None

        # ── snd_damage: low thud + noise ──
        try:
            thud_len   = int(SR * 0.08)
            noise_len  = int(SR * 0.05)
            t_thud     = np.linspace(0, 1, thud_len)
            thud       = np.sin(2 * np.pi * 55.0 * t_thud) * np.exp(-t_thud * 18) * 0.85
            noise      = np.random.uniform(-1, 1, noise_len) * np.exp(-np.linspace(0, 1, noise_len) * 10) * 0.5
            data       = np.concatenate([thud, noise])
            self.snd_damage = self._make_sound_array(data)
        except Exception as e:
            print(f"[!] snd_damage generation failed: {e}")
            self.snd_damage = None

        # ── snd_empty: dry click (very short) ──
        try:
            dur  = int(SR * 0.018)
            data = np.random.uniform(-1, 1, dur) * np.linspace(1, 0, dur) * 0.6
            self.snd_empty = self._make_sound_array(data)
        except Exception as e:
            print(f"[!] snd_empty generation failed: {e}")
            self.snd_empty = None

        # Apply master volume to all sounds
        for snd in (self.snd_shoot, self.snd_hit_tink, self.snd_reload,
                    self.snd_hit, self.snd_damage, self.snd_empty):
            if snd is not None:
                snd.set_volume(SOUND_VOLUME)

    def _play(self, snd):
        """Play a pygame Sound, silently ignoring any errors."""
        try:
            if snd is not None:
                snd.play()
        except Exception:
            pass

    # ── pre-generated visual overlays ────────────────────────

    @staticmethod
    def _make_vignette() -> pygame.Surface:
        """Dark-edge vignette, pre-rendered once."""
        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        iterations = 50
        for i in range(iterations):
            # i=0 → outermost ring (max alpha), i=49 → innermost (alpha 0)
            alpha = int(160 * (1.0 - i / iterations))
            margin = i * min(WINDOW_W, WINDOW_H) // (2 * iterations)
            rect = pygame.Rect(margin, margin,
                               WINDOW_W - 2 * margin, WINDOW_H - 2 * margin)
            ring = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            pygame.draw.rect(ring, (0, 0, 0, alpha), rect, max(1, min(WINDOW_W, WINDOW_H) // (2 * iterations) + 1))
            surf.blit(ring, (0, 0))
        return surf

    @staticmethod
    def _make_scanlines() -> pygame.Surface:
        """Subtle horizontal scanlines, pre-rendered once."""
        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for y in range(0, WINDOW_H, 3):
            pygame.draw.line(surf, (0, 0, 0, 18), (0, y), (WINDOW_W - 1, y), 1)
        return surf

    # ── serial ───────────────────────────────────────────────

    def _init_serial(self):
        try:
            self._serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
            t = threading.Thread(target=self._serial_thread, daemon=True)
            t.start()
            print(f"[OK] Serial connected on {SERIAL_PORT} @ {SERIAL_BAUD} baud")
        except Exception as e:
            print(f"[!] Serial open failed ({SERIAL_PORT}): {e}")
            print("    Check SERIAL_PORT in config and that the Pico is plugged in.")

    def _serial_thread(self):
        buf = ""
        while True:
            try:
                if self._serial_conn and self._serial_conn.in_waiting:
                    raw = self._serial_conn.read(self._serial_conn.in_waiting)
                    buf += raw.decode("utf-8", errors="ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        cmd  = line.upper()

                        if cmd in ("SHOOT", "HIT", "RELOAD"):
                            self._serial_events.put(cmd)

                        elif cmd.startswith("AIM:"):
                            # AIM:yaw,pitch  e.g. AIM:12.4,-5.1
                            try:
                                parts = line[4:].split(",")
                                yaw   = float(parts[0])
                                pitch = float(parts[1])
                                self._serial_events.put(("AIM", yaw, pitch))
                            except (ValueError, IndexError):
                                pass

                        elif cmd.startswith("JOY:"):
                            # JOY:x,y  e.g. JOY:50,-30
                            try:
                                parts = line[4:].split(",")
                                jx = float(parts[0])
                                jy = float(parts[1])
                                self._serial_events.put(("JOY", jx, jy))
                            except (ValueError, IndexError):
                                pass
                time.sleep(0.005)
            except Exception:
                time.sleep(0.1)

    # ── stream connection ────────────────────────────────────

    def _connect_stream(self):
        """Try every candidate URL in order and return the first that delivers frames."""
        if USE_WEBCAM_FALLBACK:
            print("[*] USE_WEBCAM_FALLBACK is True — using laptop webcam")
            # Try DirectShow first (more compatible on Windows), then MSMF default
            for idx in range(3):
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
                    cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            label = {cv2.CAP_DSHOW: "DirectShow",
                                     cv2.CAP_MSMF:  "MSMF"}.get(backend, "default")
                            print(f"[OK] Webcam connected (index={idx}, backend={label})")
                            self._active_url = "webcam"
                            return cap
                    cap.release()
            print("[!] No webcam found — close other apps using the camera (Teams, Discord, etc.).")
            sys.exit(1)

        print(f"\n[*] Scanning {len(STREAM_CANDIDATES)} stream URLs for {PHONE_IP} …")
        for url in STREAM_CANDIDATES:
            print(f"    Trying {url} … ", end="", flush=True)
            cap = cv2.VideoCapture(url)
            deadline = time.time() + STREAM_CONNECT_TIMEOUT
            success  = False
            while time.time() < deadline:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success = True
                        break
                time.sleep(0.15)
            if success:
                print("OK")
                print(f"[OK] Stream connected: {url}")
                self._active_url = url
                return cap
            print("failed")
            cap.release()

        print("\n[!] No working stream found. Troubleshooting steps:")
        print(f"    1. Phone and laptop on the same WiFi?")
        print(f"    2. Streaming app running and server started on your phone?")
        print(f"    3. Is PHONE_IP = '{PHONE_IP}' correct?")
        print(f"       Open your phone app and check the IP address it shows.")
        print(f"    4. Try pasting these into your laptop browser to test:")
        for url in STREAM_CANDIDATES:
            if url.startswith("http"):
                print(f"       {url}")
        print("\n    Set USE_WEBCAM_FALLBACK = True to test without a phone.")
        sys.exit(1)

    def _try_reconnect(self):
        """Attempt to reconnect to the last working URL via CameraCapture."""
        if self._capture.reconnect(self._active_url):
            print(f"[OK] Stream reconnected: {self._active_url}")

    # ── NPC tracking ─────────────────────────────────────────

    def _update_npc_tracking(self, new_boxes: list):
        """Match incoming detection boxes to existing NPC objects."""
        NPC_TIMEOUT = 0.4   # drop NPC if not seen for this many seconds (short = no ghost duplicates)
        MATCH_DIST  = 250   # max pixel distance for centre-based fallback match
        KILL_LINGER = 0.8   # seconds to show kill flash before removing NPC

        now = time.time()

        # Remove NPCs that have been dead long enough
        self.npcs = [n for n in self.npcs
                     if n.kill_time is None or (now - n.kill_time) < KILL_LINGER]
        # Remove NPCs not seen recently (only if still alive)
        self.npcs = [n for n in self.npcs
                     if n.health <= 0 or (now - n.last_seen) < NPC_TIMEOUT]

        def iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            ua = (a[2]-a[0])*(a[3]-a[1])
            ub = (b[2]-b[0])*(b[3]-b[1])
            return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0

        # Match new boxes to existing NPCs —
        # prefer IoU overlap (handles large position jumps when camera/person moves),
        # fall back to centre distance for nearby non-overlapping boxes.
        matched = set()
        for (x1, y1, x2, y2) in new_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            new_box = (x1, y1, x2, y2)

            best_idx   = None
            best_score = -1   # higher = better match
            for i, npc in enumerate(self.npcs):
                if i in matched or npc.health <= 0:
                    continue
                nbox = tuple(int(v) for v in npc.display_box)
                overlap = iou(new_box, nbox)
                if overlap > 0:
                    # Any overlap counts as a match; prefer highest IoU
                    if overlap > best_score:
                        best_score = overlap
                        best_idx   = i
                else:
                    # No overlap — use centre distance as fallback
                    nx1, ny1, nx2, ny2 = npc.box
                    d = math.hypot(cx - (nx1+nx2)//2, cy - (ny1+ny2)//2)
                    score = 1.0 - d / MATCH_DIST   # 0..1, higher = closer
                    if d < MATCH_DIST and score > best_score:
                        best_score = score
                        best_idx   = i

            if best_idx is not None:
                self.npcs[best_idx].box       = (x1, y1, x2, y2)
                self.npcs[best_idx].last_seen = now
                matched.add(best_idx)
            else:
                # Only spawn a new NPC if no existing NPC overlaps this box
                if not any(iou(new_box, tuple(int(v) for v in n.display_box)) > 0.1
                           for n in self.npcs) and len(self.npcs) < 8:
                    self.npcs.append(NPC(x1, y1, x2, y2))

    @staticmethod
    def _shrink_box(box, x_factor, y_factor=None):
        """Shrink a box inward — x_factor on left/right, y_factor on top/bottom."""
        if y_factor is None:
            y_factor = x_factor
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        dx, dy = int(w * x_factor), int(h * y_factor)
        return x1 + dx, y1 + dy, x2 - dx, y2 - dy

    def _get_targeted_npc(self):
        """Return the live NPC whose shrunk display box contains the crosshair, or None."""
        cx, cy = int(self.ch_x), int(self.ch_y)
        for npc in self.npcs:
            if npc.health <= 0:
                continue
            x1, y1, x2, y2 = self._shrink_box(
                tuple(int(v) for v in npc.display_box), HIT_SHRINK_X, HIT_SHRINK_Y)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return npc
        return None

    # ── game actions ──────────────────────────────────────────

    def shoot(self):
        now = time.time()
        if self.reloading:
            return
        if self.ammo <= 0:
            self._play(self.snd_empty)
            return
        if now - self.last_shoot < self.shoot_cooldown:
            return

        self.last_shoot = now
        self.ammo      -= 1
        self.shoot_flash = 0.08
        self.crosshair_spread = min(self.crosshair_spread + 12, 40)
        # Check if crosshair is on a live NPC
        targeted = self._get_targeted_npc()
        if targeted:
            # ── Confirmed NPC hit — real gunshot + impact tink overlay ──
            self._play(self.snd_shoot)
            self._play(self.snd_hit_tink)
            damage = 25
            targeted.health    = max(0, targeted.health - damage)
            targeted.hit_flash = 0.25
            self.hit_markers.append(
                HitMarker(int(self.ch_x), int(self.ch_y), confirmed=True))
            self.score += 50

            if targeted.health <= 0:
                targeted.kill_time = time.time()
                self.kills += 1
                self.score += 150   # kill bonus
                self._play(self.snd_hit)
                self.kill_feed.append([time.time(), "HOSTILE ELIMINATED"])
        else:
            # ── Miss — standard gunshot only ──
            self._play(self.snd_shoot)
            self.hit_markers.append(
                HitMarker(int(self.ch_x), int(self.ch_y), confirmed=False))
            # No score for a miss (encourage aiming)

        if self.ammo == 0:
            self.start_reload()

    def start_reload(self):
        if not self.reloading and self.ammo < self.max_ammo:
            self.reloading    = True
            self.reload_start = time.time()
            self._play(self.snd_reload)

    def take_damage(self, amount=15):
        self.health      = max(0, self.health - amount)
        self.damage_flash = 0.3
        self._play(self.snd_damage)
        if self.health == 0:
            self.game_active = False

    def reset(self):
        self.score       = 0
        self.health      = 100
        self.ammo        = self.max_ammo
        self.kills       = 0
        self.reloading   = False
        self.game_active = True
        self.hit_markers = []
        self.npcs        = []

    # ── update ────────────────────────────────────────────────

    def update(self, dt):
        # Crosshair spread decay
        self.crosshair_spread = max(0, self.crosshair_spread - 30 * dt)

        # Flash timers
        self.damage_flash = max(0, self.damage_flash - dt)
        self.shoot_flash  = max(0, self.shoot_flash  - dt)

        # Reload
        if self.reloading:
            if time.time() - self.reload_start >= self.reload_time:
                self.ammo      = self.max_ammo
                self.reloading = False

        # Hit markers
        self.hit_markers = [h for h in self.hit_markers if h.alive()]

        # FPS counter
        self.frame_times.append(time.time())
        self.frame_times = [t for t in self.frame_times if time.time() - t < 1.0]
        self.fps_display = len(self.frame_times)

        # NPC hit-flash timers + smooth display_box toward real box
        lerp = min(1.0, BOX_SMOOTH_SPEED * dt)
        for npc in self.npcs:
            npc.hit_flash = max(0.0, npc.hit_flash - dt)
            dx1, dy1, dx2, dy2 = npc.display_box
            tx1, ty1, tx2, ty2 = npc.box
            npc.display_box = (
                dx1 + (tx1 - dx1) * lerp,
                dy1 + (ty1 - dy1) * lerp,
                dx2 + (tx2 - dx2) * lerp,
                dy2 + (ty2 - dy2) * lerp,
            )
            # Smooth display_health toward real health at 80 HP/s (visibly drains)
            npc.display_health = max(float(npc.health),
                                     npc.display_health - 80.0 * dt)

        # Pull latest detections from background worker
        if self._det_worker is not None:
            new_boxes = self._det_worker.get_detections()
            self._update_npc_tracking(new_boxes)

        # Process Pico 1 serial events
        while True:
            try:
                event = self._serial_events.get_nowait()
            except queue.Empty:
                break

            # AIM tuple — update crosshair from IMU angles
            if isinstance(event, tuple) and event[0] == "AIM":
                _, yaw, pitch = event
                # Map yaw  ±AIM_YAW_RANGE   → screen X 0..WINDOW_W
                #     pitch ±AIM_PITCH_RANGE → screen Y 0..WINDOW_H
                target_x = (yaw   / AIM_YAW_RANGE  + 1.0) * 0.5 * WINDOW_W
                target_y = (pitch / AIM_PITCH_RANGE + 1.0) * 0.5 * WINDOW_H
                target_x = max(0.0, min(float(WINDOW_W), target_x))
                target_y = max(0.0, min(float(WINDOW_H), target_y))
                s = AIM_SMOOTHING
                self.ch_x = self.ch_x * s + target_x * (1.0 - s)
                self.ch_y = self.ch_y * s + target_y * (1.0 - s)
                continue

            # JOY tuple — joystick controls robot movement via Pico nRF24,
            # nothing for the laptop game to do with it
            if isinstance(event, tuple) and event[0] == "JOY":
                continue

            # String events
            cmd = event
            if not self.game_active:
                continue
            if cmd == "SHOOT":
                self.shoot()
            elif cmd == "HIT":
                self.take_damage()
            elif cmd == "RELOAD":
                self.start_reload()

    # ── video frame ──────────────────────────────────────────

    def get_frame(self):
        frame = self._capture.get_latest()

        if frame is None:
            # No new frame yet — return the last good surface so the screen
            # doesn't flash black during brief network hiccups.
            # If the stream has been silent for >2 s, trigger a reconnect
            # regardless of whether we have a cached surface.
            now   = time.time()
            stale = (now - self._capture.last_frame_time) > 2.0
            if stale and now >= self._reconnect_after:
                self._reconnect_after = now + 3.0
                threading.Thread(target=self._try_reconnect, daemon=True).start()
            return self._last_surface

        frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

        # Submit BGR frame to detection worker (non-blocking, drops if busy)
        if self._det_worker is not None:
            self._det_worker.submit(frame)

        # ── cinematic color grade ──────────────────────────────────────────
        # Boost contrast, slight desaturation, cool blue tint — makes the
        # raw camera feed look like a game render rather than plain video.
        frame = cv2.convertScaleAbs(frame, alpha=1.25, beta=-18)   # contrast + lift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.72          # desaturate to ~72 % of original
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + 12, 0, 255)  # blue +12
        frame[:, :, 2] = np.clip(frame[:, :, 2].astype(np.int16) - 8,  0, 255)  # red  -8

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        surface = pygame.surfarray.make_surface(frame)
        surface = pygame.transform.flip(surface, True, False)
        self._last_surface = surface
        return surface

    # ── draw helpers ─────────────────────────────────────────

    def draw_crosshair(self):
        cx, cy = int(self.ch_x), int(self.ch_y)
        s   = int(self.crosshair_spread)
        gap = 6 + s
        size  = 14
        thick = 2

        # Tint red when a target is in crosshair
        targeted = self._get_targeted_npc()
        col = (255, 60, 60) if targeted else CROSSHAIR_C

        shadow = (0, 0, 0)
        for ox, oy in [(1, 1)]:
            pygame.draw.line(self.screen, shadow, (cx+gap+ox, cy+oy), (cx+gap+size+ox, cy+oy), thick)
            pygame.draw.line(self.screen, shadow, (cx-gap+ox, cy+oy), (cx-gap-size+ox, cy+oy), thick)
            pygame.draw.line(self.screen, shadow, (cx+ox, cy+gap+oy), (cx+ox, cy+gap+size+oy), thick)
            pygame.draw.line(self.screen, shadow, (cx+ox, cy-gap+oy), (cx+ox, cy-gap-size+oy), thick)

        pygame.draw.line(self.screen, col, (cx+gap, cy), (cx+gap+size, cy), thick)
        pygame.draw.line(self.screen, col, (cx-gap, cy), (cx-gap-size, cy), thick)
        pygame.draw.line(self.screen, col, (cx, cy+gap), (cx, cy+gap+size), thick)
        pygame.draw.line(self.screen, col, (cx, cy-gap), (cx, cy-gap-size), thick)
        pygame.draw.circle(self.screen, col, (cx, cy), 2)

    def draw_hit_markers(self):
        for hm in self.hit_markers:
            a    = max(0.0, min(1.0, hm.alpha()))   # clamp to avoid negative RGB
            size = 14 if hm.confirmed else 10
            if hm.confirmed:
                col = (255, int(50 * a), int(50 * a))   # red — confirmed hit
            else:
                col = (200, int(200 * a), int(200 * a)) # grey — miss
            thick = 2
            cx, cy = hm.x, hm.y
            pygame.draw.line(self.screen, col,
                (cx-size, cy-size), (cx-size//2, cy-size//2), thick)
            pygame.draw.line(self.screen, col,
                (cx+size, cy-size), (cx+size//2, cy-size//2), thick)
            pygame.draw.line(self.screen, col,
                (cx-size, cy+size), (cx-size//2, cy+size//2), thick)
            pygame.draw.line(self.screen, col,
                (cx+size, cy+size), (cx+size//2, cy+size//2), thick)

    def draw_npc_overlays(self):
        """Draw a slim health bar above each detected target. No brackets or labels."""
        targeted = self._get_targeted_npc()
        for npc in self.npcs:
            if npc.health <= 0:
                continue

            x1, y1, x2, _  = (int(v) for v in npc.display_box)
            w = x2 - x1
            if w <= 0:
                continue

            pct     = npc.display_health / npc.MAX_HEALTH
            bar_h   = 5
            bar_y   = y1 - bar_h - 4

            # Bar colour: red when crosshair is on this target, else green→orange→red by HP
            if npc is targeted:
                bar_col = (220, 30, 30)
            elif npc.hit_flash > 0:
                bar_col = (255, 255, 255)
            elif pct > 0.5:
                bar_col = (0, 210, 80)
            elif pct > 0.25:
                bar_col = (255, 140, 0)
            else:
                bar_col = (220, 30, 30)

            # Dark background track
            pygame.draw.rect(self.screen, (15, 15, 15, 180), (x1, bar_y, w, bar_h))
            # Filled portion
            pygame.draw.rect(self.screen, bar_col, (x1, bar_y, max(1, int(w * pct)), bar_h))

    def draw_health_bar(self):
        x, y = 40, WINDOW_H - 80
        w, h = 220, 18
        pct  = self.health / 100

        pygame.draw.rect(self.screen, (30, 30, 30), (x-2, y-2, w+4, h+4))
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, w, h))

        col = GREEN if pct > 0.5 else ORANGE if pct > 0.25 else RED
        pygame.draw.rect(self.screen, col, (x, y, int(w * pct), h))
        pygame.draw.rect(self.screen, HUD_GREEN, (x-2, y-2, w+4, h+4), 1)

        label = self.font_small.render(f"HEALTH  {self.health}%", True, HUD_GREEN)
        self.screen.blit(label, (x, y - 20))
        icon = self.font_med.render("♥", True, col)
        self.screen.blit(icon, (x + w + 10, y - 4))

    def draw_ammo(self):
        x, y = WINDOW_W - 200, WINDOW_H - 80
        label = self.font_small.render("AMMO", True, HUD_DIM)
        self.screen.blit(label, (x, y - 20))

        if self.reloading:
            progress = (time.time() - self.reload_start) / self.reload_time
            w, h = 160, 10
            pygame.draw.rect(self.screen, (30, 30, 30), (x, y + 8, w, h))
            pygame.draw.rect(self.screen, ORANGE, (x, y + 8, int(w * progress), h))
            pygame.draw.rect(self.screen, HUD_GREEN, (x, y + 8, w, h), 1)
            txt = self.font_med.render("RELOADING...", True, ORANGE)
            self.screen.blit(txt, (x, y - 4))
        else:
            pip_w, pip_h = 8, 16
            gap = 4
            for i in range(self.max_ammo):
                col = HUD_GREEN if i < self.ammo else (40, 40, 40)
                bx  = x + i * (pip_w + gap)
                if i >= 15:
                    bx = x + (i - 15) * (pip_w + gap)
                    by = y + pip_h + 4
                else:
                    by = y
                pygame.draw.rect(self.screen, col, (bx, by, pip_w, pip_h))

            count = self.font_hud.render(f"{self.ammo:02d}/{self.max_ammo}", True, HUD_GREEN)
            self.screen.blit(count, (x, y - 4))

    def draw_score(self):
        x = WINDOW_W // 2
        txt = self.font_hud.render(f"SCORE  {self.score:06d}", True, HUD_GREEN)
        self.screen.blit(txt, txt.get_rect(centerx=x).move(0, 20))

        kills_txt = self.font_small.render(f"KILLS: {self.kills}", True, HUD_DIM)
        self.screen.blit(kills_txt, kills_txt.get_rect(centerx=x).move(0, 60))

    def draw_kill_feed(self):
        """Draw the last 4 kills on the right side, fading over 4 seconds."""
        FADE_TIME = 4.0
        now = time.time()
        # Prune old entries
        self.kill_feed = [e for e in self.kill_feed if now - e[0] < FADE_TIME]
        recent = self.kill_feed[-4:]   # show at most 4
        x = WINDOW_W - 260
        y = 90
        for entry in reversed(recent):
            age   = now - entry[0]
            frac  = max(0.0, 1.0 - age / FADE_TIME)
            green = int(200 * frac)
            col   = (0, green, int(60 * frac))
            txt   = self.font_med.render(entry[1], True, col)
            self.screen.blit(txt, (x, y))
            y += txt.get_height() + 4

    def draw_fps(self):
        txt = self.font_small.render(f"FPS {self.fps_display}", True, (80, 80, 80))
        self.screen.blit(txt, (WINDOW_W - 80, 10))

    def draw_corner_brackets(self):
        col    = HUD_GREEN
        size   = 30
        thick  = 2
        margin = 20
        for bx, by in [
                (margin, margin),
                (WINDOW_W - margin, margin),
                (margin, WINDOW_H - margin),
                (WINDOW_W - margin, WINDOW_H - margin)]:
            dx = 1 if bx == margin else -1
            dy = 1 if by == margin else -1
            pygame.draw.line(self.screen, col,
                (bx, by), (bx + dx*size, by), thick)
            pygame.draw.line(self.screen, col,
                (bx, by), (bx, by + dy*size), thick)

    def draw_damage_flash(self):
        if self.damage_flash > 0:
            alpha   = int(180 * (self.damage_flash / 0.3))
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((180, 0, 0, alpha))
            self.screen.blit(overlay, (0, 0))
            for i in range(8):
                a = int(100 * (self.damage_flash / 0.3) * (1 - i / 8))
                border_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
                pygame.draw.rect(border_surf, (200, 0, 0, a),
                    (i*3, i*3, WINDOW_W - i*6, WINDOW_H - i*6), 6)
                self.screen.blit(border_surf, (0, 0))

    def draw_shoot_flash(self):
        if self.shoot_flash > 0:
            alpha   = int(60 * (self.shoot_flash / 0.08))
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((255, 255, 200, alpha))
            self.screen.blit(overlay, (0, 0))

    def draw_reload_hint(self):
        if self.ammo < 8 and not self.reloading:
            pulse = abs(math.sin(time.time() * 4))
            col   = (255, int(140 * pulse), 0)
            txt   = self.font_med.render("[ R ]  RELOAD", True, col)
            self.screen.blit(txt,
                txt.get_rect(centerx=WINDOW_W // 2).move(0, WINDOW_H // 2 + 60))

    def draw_hud_title(self):
        txt = self.font_small.render("ROBO-HUNTER  //  HACKABOT 2026", True, (50, 100, 60))
        self.screen.blit(txt, (40, 14))

    def draw_serial_status(self):
        if ENABLE_SERIAL:
            status = "[PICO CONNECTED]" if self._serial_conn else "[PICO OFFLINE]"
            col    = HUD_DIM if self._serial_conn else (100, 40, 40)
            txt    = self.font_small.render(status, True, col)
            self.screen.blit(txt, (WINDOW_W - txt.get_width() - 10, WINDOW_H - 24))

    def draw_game_over(self):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        title = self.font_big.render("GAME OVER", True, RED)
        self.screen.blit(title, title.get_rect(center=(WINDOW_W//2, WINDOW_H//2 - 40)))

        score_txt = self.font_hud.render(f"FINAL SCORE: {self.score}", True, HUD_GREEN)
        self.screen.blit(score_txt, score_txt.get_rect(center=(WINDOW_W//2, WINDOW_H//2 + 20)))

        hint = self.font_small.render("PRESS R TO RESTART", True, WHITE)
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_W//2, WINDOW_H//2 + 70)))

    # ── main loop ────────────────────────────────────────────

    def run(self):
        running   = True
        prev_time = time.time()

        print("[*] Game running!")
        print("    SPACE or LEFT CLICK  = shoot")
        print("    R                    = reload")
        print("    D                    = simulate taking damage")
        print("    ESC                  = quit")
        if ENABLE_SERIAL:
            print(f"    Pico 1 serial        = {SERIAL_PORT} @ {SERIAL_BAUD} baud")
        if ENABLE_DETECTION:
            method = "YOLOv8n" if (USE_YOLO and YOLO_AVAILABLE) else "HOG"
            print(f"    Detection method     = {method}")

        while running:
            now      = time.time()
            dt       = now - prev_time
            prev_time = now

            # ── EVENTS ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        if not self.game_active:
                            self.reset()
                        else:
                            self.start_reload()
                    if event.key == pygame.K_SPACE and self.game_active:
                        self.shoot()
                    if event.key == pygame.K_d and self.game_active:
                        self.take_damage()

                # Mouse moves crosshair when gun serial is not connected (testing)
                if event.type == pygame.MOUSEMOTION and not ENABLE_SERIAL:
                    self.ch_x = float(event.pos[0])
                    self.ch_y = float(event.pos[1])

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.game_active:
                        self.shoot()

            # ── UPDATE ──────────────────────────────────────
            self.update(dt)

            # ── DRAW ────────────────────────────────────────
            # 1. Video frame
            frame = self.get_frame()
            if frame:
                self.screen.blit(frame, (0, 0))
            else:
                self.screen.fill((10, 10, 10))
                err = self.font_med.render("NO VIDEO SIGNAL", True, RED)
                self.screen.blit(err, err.get_rect(center=(WINDOW_W//2, WINDOW_H//2)))

            # 2. NPC detection overlays (below HUD, above raw video)
            self.draw_npc_overlays()

            # 3. Shoot flash (muzzle effect, before HUD)
            self.draw_shoot_flash()

            # 4. Vignette + scanlines (atmospheric, under HUD text)
            self.screen.blit(self._vignette, (0, 0))
            self.screen.blit(self._scanlines, (0, 0))

            # 5. HUD
            self.draw_corner_brackets()
            self.draw_hud_title()
            self.draw_crosshair()
            self.draw_hit_markers()
            self.draw_ammo()
            self.draw_score()
            self.draw_kill_feed()
            self.draw_fps()
            self.draw_reload_hint()
            self.draw_serial_status()

            # 6. Damage flash (on top of everything)
            self.draw_damage_flash()

            # 7. Game over screen
            if not self.game_active:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(60)

        self._capture.release()
        pygame.quit()
        print("[*] Game closed.")


if __name__ == "__main__":
    game = FPSGame()
    game.run()
