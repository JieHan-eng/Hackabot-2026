import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
import threading
import queue
import os

# Absolute path to the project folder — used for sound/music file loading
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

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
USE_WEBCAM_FALLBACK = False

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

# True  = launch in fullscreen (resolution auto-detected)
# False = windowed 1280×720
# F11   = toggle at any time while the game is running
FULLSCREEN = True

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
#  SOUND_VOLUME  : volume for gunshot / hit / damage sounds  (0.0 – 1.0)
#  RELOAD_VOLUME : volume for the reload sound               (0.0 – 1.0)
#                  Set this higher than SOUND_VOLUME if the reload feels too quiet.
#
SOUND_VOLUME  = 0.8
RELOAD_VOLUME = 1.0   # ← raise this if reload is still too soft
MUSIC_VOLUME  = 0.4   # background music volume (separate from SFX)

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
ENABLE_SERIAL = False     # set False to use mouse aim instead
SERIAL_PORT   = "AUTO"    # "AUTO" = find Pico automatically, or set e.g. "COM4"
SERIAL_BAUD   = 115200

# ─── AIM / CROSSHAIR CONFIG ───────────────────────────────────
#
#  AIM_YAW_RANGE   : gun yaw angle (degrees) that maps to screen edge left/right
#                    e.g. 45 means ±45° of yaw covers the full screen width
#  AIM_PITCH_RANGE : same but for up/down
#  AIM_SMOOTHING   : 0.0 = raw (jittery), 1.0 = never moves. 0.25 is a good start.
#  AIM_YAW_INVERT  : set True if turning gun right moves crosshair LEFT
#  AIM_PITCH_INVERT: set True if aiming gun UP moves crosshair DOWN
#
AIM_YAW_RANGE    = 45.0    # degrees — tune to how wide your physical sweep is
AIM_PITCH_RANGE  = 20.0    # degrees
AIM_SMOOTHING    = 0.25    # low-pass smoothing on crosshair position (0 = off)
AIM_YAW_INVERT   = False   # flip left/right if crosshair goes wrong way
AIM_PITCH_INVERT = False   # flip up/down   if crosshair goes wrong way

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

# Per-map data: name, subtitle, tags, accent_colour, bg_colour, particle_colour
MAP_DATA = [
    {
        "name":     "ARCTIC STORM",
        "sub":      "FROZEN TUNDRA  •  BLIZZARD CONDITIONS",
        "tags":     ["SNOW", "WIND", "ZERO VISIBILITY"],
        "accent":   (140, 200, 255),
        "bg":       (8,  18,  35),
        "particle": (200, 220, 255),
        "filter":   "arctic",
    },
    {
        "name":     "WARZONE",
        "sub":      "URBAN RUINS  •  ACTIVE COMBAT ZONE",
        "tags":     ["FIRE", "SMOKE", "URBAN"],
        "accent":   (220, 100, 30),
        "bg":       (25,  8,   5),
        "particle": (255, 130, 40),
        "filter":   "warzone",
    },
    {
        "name":     "JUNGLE WARFARE",
        "sub":      "DENSE CANOPY  •  HIGH HUMIDITY",
        "tags":     ["RAIN", "FOG", "VEGETATION"],
        "accent":   (60,  200, 80),
        "bg":       (5,   18,  8),
        "particle": (100, 200, 80),
        "filter":   "jungle",
    },
    {
        "name":     "NIGHT OPS",
        "sub":      "CLASSIFIED FACILITY  •  LIGHTS OUT",
        "tags":     ["DARKNESS", "STEALTH", "NV ACTIVE"],
        "accent":   (40,  220, 130),
        "bg":       (5,   10,  22),
        "particle": (60,  180, 100),
        "filter":   "nightops",
    },
    {
        "name":     "CYBERPUNK CITY",
        "sub":      "NEON DISTRICT  •  2087",
        "tags":     ["NEON", "RAIN", "URBAN"],
        "accent":   (0,   210, 255),
        "bg":       (8,   3,   25),
        "particle": (200, 0,   255),
        "filter":   "cyberpunk",
    },
]
MAP_NAMES = [m["name"] for m in MAP_DATA]


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
        self.dead_until     = None  # timestamp when this NPC may respawn (None = alive)
        self.kill_icon_until = None # timestamp until skull icon is shown
        self.engaged        = False # True once this NPC has been hit at least once


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

    def reconnect_cap(self, cap: cv2.VideoCapture):
        """Directly swap in an already-opened VideoCapture."""
        self._swap(cap)

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
        global WINDOW_W, WINDOW_H
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()
        pygame.display.set_caption("ROBO-HUNTER // HACKABOT 2026")
        if FULLSCREEN:
            info = pygame.display.Info()
            WINDOW_W = info.current_w
            WINDOW_H = info.current_h
            self.screen = pygame.display.set_mode(
                (WINDOW_W, WINDOW_H), pygame.FULLSCREEN)
        else:
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

        # ── State machine ──
        # Start at CONNECTING only when using phone stream; webcam goes straight to TITLE
        self.state          = "CONNECTING" if not USE_WEBCAM_FALLBACK else "TITLE"
        self.current_map    = 0
        self._title_timer   = 0.0
        self._calib_timer   = 0.0
        self._calib_delay   = 0.5        # seconds before bar starts filling
        self._calib_duration = 3.0       # calibration bar fill time
        self._countdown_val   = 3
        self._countdown_timer = 0.0
        self._music_track   = None
        self._particles: list = []

        # Gun animation state (used for shake/flash even without gun model)
        self.gun_recoil   = 0.0
        self.gun_bob      = 0.0
        self.gun_reload_y = 0.0
        self.shake_x      = 0.0
        self.shake_y      = 0.0
        self.shake_time   = 0.0

        # Volume slider
        self._current_volume = SOUND_VOLUME
        self._vol_display_until = 0.0
        self._vol_dragging   = False

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
        # Start menu music immediately on launch
        self._set_music("menu")

        # ── Pre-generated visual overlays ──
        self._vignette     = self._make_vignette()
        self._scanlines    = self._make_scanlines()
        self._grain_frames = self._make_grain_frames()
        self._grain_idx    = 0
        self._grain_tick   = 0.0
        self._nv_mask      = self._make_nv_mask()

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
        self._dbg_yaw   = 0.0   # last received yaw (for debug display)
        self._dbg_pitch = 0.0   # last received pitch (for debug display)
        self._aim_yaw_offset   = 0.0  # subtracted from all AIM yaw readings
        self._aim_pitch_offset = 0.0  # subtracted from all AIM pitch readings
        self._rezero_flash     = 0.0  # seconds remaining for "AIM ZEROED" flash
        if ENABLE_SERIAL and SERIAL_AVAILABLE:
            self._init_serial()
        elif ENABLE_SERIAL and not SERIAL_AVAILABLE:
            print("[!] ENABLE_SERIAL=True but pyserial is not installed")

        # Video stream — connect in background so window opens immediately
        self._capture         = CameraCapture(cv2.VideoCapture())  # blank until connected
        self._last_surface    = None
        self._reconnect_after = 0.0
        self._stream_connecting = True
        threading.Thread(target=self._connect_stream_bg, daemon=True).start()

    # ── procedural sound generation ──────────────────────────

    @staticmethod
    def _make_sound_array(samples: np.ndarray) -> pygame.mixer.Sound:
        """Convert a mono float32 array (-1..1) to a stereo int16 pygame Sound."""
        clipped = np.clip(samples, -1.0, 1.0)
        mono    = (clipped * 32767).astype(np.int16)
        stereo  = np.column_stack([mono, mono])
        return pygame.sndarray.make_sound(stereo)

    @staticmethod
    def _load_wav(path: str, trim_s: float = None, amplify: float = 1.0) -> pygame.mixer.Sound:
        """Load a WAV file as a pygame Sound. pygame handles sample-rate conversion.
        trim_s: if set, truncate to this many seconds.
        amplify: multiply sample data by this factor (>1.0 makes it louder than set_volume allows)."""
        snd = pygame.mixer.Sound(path)
        if trim_s is not None or amplify != 1.0:
            arr = pygame.sndarray.array(snd).astype(np.float32)
            if trim_s is not None:
                mixer_freq, _, _ = pygame.mixer.get_init()
                n = int(trim_s * mixer_freq)
                if len(arr) > n:
                    arr = arr[:n]
            if amplify != 1.0:
                arr = np.clip(arr * amplify, -32768, 32767)
            snd = pygame.sndarray.make_sound(np.ascontiguousarray(arr.astype(np.int16)))
        return snd

    def _init_sounds(self):
        SR = 44100  # sample rate

        # ── snd_shoot: real revolver WAV, trimmed to first shot (~0.25 s) ──
        try:
            self.snd_shoot = self._load_wav(
                os.path.join(_ASSETS_DIR, "revolver_shot.wav"), trim_s=0.25)
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

        # ── snd_reload: revolver reload sound (tries mp3, then wav variants) ──
        try:
            _rp = os.path.join(_ASSETS_DIR, "revolver_reload.mp3")
            if not os.path.exists(_rp):
                _rp = os.path.join(_ASSETS_DIR, "revolver_reload.wav")
            if not os.path.exists(_rp):
                _rp = os.path.join(_ASSETS_DIR, "revolver reload.wav")
            if not os.path.exists(_rp):
                raise FileNotFoundError(f"No reload sound found in {_ASSETS_DIR}")
            self.snd_reload = self._load_wav(_rp, amplify=2.5)
            print(f"[OK] Reload sound loaded: {os.path.basename(_rp)}")
        except Exception as e:
            print(f"[!] snd_reload failed: {e}")
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

        # Apply volumes — reload gets its own setting (often needs to be louder)
        for snd in (self.snd_shoot, self.snd_hit_tink,
                    self.snd_hit, self.snd_damage, self.snd_empty):
            if snd is not None:
                snd.set_volume(SOUND_VOLUME)
        if self.snd_reload is not None:
            self.snd_reload.set_volume(RELOAD_VOLUME)

    def _set_volume(self, vol: float):
        """Set master volume for all sounds + music and show the HUD indicator."""
        self._current_volume = round(vol, 1)
        for snd in (self.snd_shoot, self.snd_hit_tink,
                    self.snd_hit, self.snd_damage, self.snd_empty):
            if snd is not None:
                snd.set_volume(self._current_volume)
        if self.snd_reload is not None:
            self.snd_reload.set_volume(min(1.0, self._current_volume))
        # Scale music proportionally to slider (music sits lower than SFX)
        try:
            pygame.mixer.music.set_volume(self._current_volume * MUSIC_VOLUME)
        except Exception:
            pass
        self._vol_display_until = time.time() + 2.0

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
        """CRT-style horizontal scanlines, pre-rendered once."""
        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for y in range(0, WINDOW_H, 4):
            pygame.draw.line(surf, (0, 0, 0, 30), (0, y), (WINDOW_W - 1, y), 1)
        for y in range(0, WINDOW_H, 16):
            pygame.draw.line(surf, (0, 0, 0, 12), (0, y), (WINDOW_W - 1, y), 2)
        return surf

    @staticmethod
    def _make_grain_frames(n: int = 8) -> list:
        """Pre-generate n random noise surfaces for film grain cycling."""
        frames = []
        for _ in range(n):
            surf = pygame.Surface((WINDOW_W, WINDOW_H))
            arr  = pygame.surfarray.pixels3d(surf)
            arr[:] = np.random.randint(80, 170, (WINDOW_W, WINDOW_H, 3), dtype=np.uint8)
            del arr
            surf.set_alpha(22)
            frames.append(surf)
        return frames

    @staticmethod
    def _make_nv_mask() -> pygame.Surface:
        """Night-vision goggle scope mask — black outside a central oval."""
        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 255))
        # Cut out ellipse in the center
        rx, ry = int(WINDOW_W * 0.38), int(WINDOW_H * 0.48)
        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        # Soft edge: draw concentric shrinking ellipses from black→transparent
        for i in range(40):
            a = int(255 * (i / 40))   # alpha goes 0→255 outward; we build inward
            er = max(1, rx - i * (rx // 40))
            ep = max(1, ry - i * (ry // 40))
            pygame.draw.ellipse(surf, (0, 0, 0, 255 - a),
                                (cx - er, cy - ep, er * 2, ep * 2), 0)
        return surf

    # ── grain / particles / overlays ─────────────────────────

    def _draw_grain(self, dt: float):
        """Cycle through pre-baked noise frames for film-grain texture."""
        self._grain_tick += dt
        if self._grain_tick > 0.08:
            self._grain_tick = 0.0
            self._grain_idx = (self._grain_idx + 1) % len(self._grain_frames)
        self.screen.blit(self._grain_frames[self._grain_idx], (0, 0))

    def _update_and_draw_particles(self, dt: float, map_idx: int):
        """Spawn + update + draw ambient menu particles themed to the current map."""
        f = MAP_DATA[map_idx]["filter"] if map_idx >= 0 else "nightops"
        pc = MAP_DATA[map_idx]["particle"] if map_idx >= 0 else (60, 180, 100)

        # Spawn rate and behaviour per map
        if f == "arctic":        # snowflakes
            if random.random() < 0.4:
                self._particles.append([
                    random.randint(0, WINDOW_W), -4,
                    random.uniform(-0.6, 0.6), random.uniform(0.6, 1.4),
                    random.uniform(1.5, 3.5), random.uniform(1.5, 3.5),
                    random.randint(2, 4), *pc])
        elif f == "warzone":     # rising embers
            if random.random() < 0.35:
                self._particles.append([
                    random.randint(0, WINDOW_W), WINDOW_H + 4,
                    random.uniform(-0.5, 0.5), random.uniform(-1.8, -0.8),
                    random.uniform(1.0, 2.5), random.uniform(1.0, 2.5),
                    random.randint(2, 3), *pc])
        elif f == "jungle":      # falling rain streaks
            if random.random() < 0.7:
                self._particles.append([
                    random.randint(0, WINDOW_W), -8,
                    random.uniform(0.3, 0.8), random.uniform(3.0, 5.0),
                    random.uniform(0.6, 1.2), random.uniform(0.6, 1.2),
                    1, *pc])
        elif f == "nightops":    # slow drifting specks
            if random.random() < 0.15:
                self._particles.append([
                    random.randint(0, WINDOW_W), random.randint(0, WINDOW_H),
                    random.uniform(-0.2, 0.2), random.uniform(-0.4, -0.1),
                    random.uniform(3.0, 6.0), random.uniform(3.0, 6.0),
                    random.randint(1, 2), *pc])
        else:                    # cyberpunk — vertical data rain
            if random.random() < 0.5:
                self._particles.append([
                    random.randint(0, WINDOW_W), -10,
                    0.0, random.uniform(4.0, 9.0),
                    random.uniform(0.3, 0.8), random.uniform(0.3, 0.8),
                    1, *pc])

        # Update + draw
        alive = []
        for p in self._particles:
            p[0] += p[2] * dt * 60
            p[1] += p[3] * dt * 60
            p[4] -= dt
            if p[4] <= 0:
                continue
            alpha = min(255, max(0, int(180 * (p[4] / p[5]))))
            r, g, b = p[7], p[8], p[9]
            size = p[6]
            if f == "jungle":   # draw as short line streak
                pygame.draw.line(self.screen, (r, g, b),
                                 (int(p[0]), int(p[1])),
                                 (int(p[0] + p[2]*4), int(p[1] + p[3]*4)), 1)
            elif f == "cyberpunk":  # single bright pixel
                col = (min(255, r + 80), min(255, g + 80), min(255, b + 80))
                self.screen.set_at((int(p[0]) % WINDOW_W, int(p[1]) % WINDOW_H), col)
            else:
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (r, g, b, alpha), (size, size), size)
                self.screen.blit(surf, (int(p[0]) - size, int(p[1]) - size))
            alive.append(p)
        self._particles = alive[:300]   # cap list size

    def _draw_map_terrain(self, rx: int, ry: int, rw: int, rh: int,
                          map_filter: str, accent: tuple):
        """Draw a procedural map-themed terrain silhouette.
        Uses a seeded RNG so the layout is stable across frames."""
        rng    = random.Random(map_filter)   # deterministic per map
        base_y = ry + rh
        col    = (int(accent[0]*0.20), int(accent[1]*0.20), int(accent[2]*0.20))
        col2   = (int(accent[0]*0.10), int(accent[1]*0.10), int(accent[2]*0.10))

        if map_filter == "arctic":
            pts = [(rx, base_y)]
            x = rx
            while x < rx + rw:
                pts.append((x, base_y - rng.randint(30, 160)))
                x += rng.randint(30, 80)
            pts.append((rx + rw, base_y))
            pygame.draw.polygon(self.screen, col, pts)
            for i in range(1, len(pts) - 1):
                px, py = pts[i]
                if py < base_y - 80:
                    pygame.draw.polygon(self.screen, (200, 220, 255),
                                        [(px-12, py+25), (px, py), (px+12, py+25)])

        elif map_filter == "warzone":
            x = rx
            while x < rx + rw:
                bw = rng.randint(25, 65)
                bh = rng.randint(40, 180)
                notch = rng.randint(0, 2)
                pts = [(x, base_y), (x, base_y - bh)]
                if notch:
                    pts += [(x + bw//3, base_y - bh + rng.randint(8, 20)),
                            (x + 2*bw//3, base_y - bh)]
                pts += [(x + bw, base_y - bh + rng.randint(0, 15)), (x + bw, base_y)]
                pygame.draw.polygon(self.screen, col, pts)
                for wy in range(int(base_y - bh + 12), base_y - 10, 18):
                    if rng.random() > 0.5:
                        pygame.draw.rect(self.screen,
                                         (int(accent[0]*0.4), int(accent[1]*0.4), 0),
                                         (x + 6, wy, 6, 8))
                x += bw + rng.randint(2, 10)

        elif map_filter == "jungle":
            x = rx
            while x < rx + rw:
                tx = x + rng.randint(-10, 10)
                tr = rng.randint(25, 60)
                ty = base_y - rng.randint(60, 140)
                pygame.draw.circle(self.screen, col, (tx, ty), tr)
                pygame.draw.circle(self.screen, col2, (tx - 8, ty + 10), max(1, tr - 10))
                pygame.draw.rect(self.screen, col2, (tx - 4, ty + tr - 10, 8, 50))
                x += rng.randint(40, 80)
            pygame.draw.rect(self.screen, col2, (rx, base_y - 15, rw, 15))

        elif map_filter == "nightops":
            pts = [(rx, base_y)]
            steps = rw // 8
            for i in range(steps + 1):
                wave = math.sin(i * 0.4) * 35 + math.sin(i * 0.9) * 15
                pts.append((rx + i * 8, base_y - 40 - int(wave)))
            pts.append((rx + rw, base_y))
            pygame.draw.polygon(self.screen, col, pts)
            for _ in range(6):
                tx = rng.randint(rx + 20, rx + rw - 20)
                th = rng.randint(40, 80)
                ty_base = base_y - 38
                for level in range(3):
                    lw = th // 2 - level * 8
                    ly = ty_base - th + level * (th // 3)
                    pygame.draw.polygon(self.screen, col2,
                                        [(tx, ly), (tx-lw, ly+th//3), (tx+lw, ly+th//3)])

        else:  # cyberpunk
            x = rx
            while x < rx + rw:
                bw = rng.randint(30, 70)
                bh = rng.randint(80, 220)
                pygame.draw.rect(self.screen, col, (x, base_y - bh, bw, bh))
                ax = x + bw // 2
                pygame.draw.line(self.screen, col2,
                                 (ax, base_y - bh), (ax, base_y - bh - 30), 2)
                for wy in range(int(base_y - bh + 8), base_y - 4, 12):
                    pygame.draw.rect(self.screen,
                                     (0, int(accent[1]*0.5), int(accent[2]*0.5)),
                                     (x + 4, wy, bw - 8, 4))
                x += bw + rng.randint(4, 16)

    def draw_map_overlay(self, dt: float):
        """Draw per-map in-game atmosphere overlay on top of the camera feed."""
        f   = MAP_DATA[self.current_map]["filter"]
        acc = MAP_DATA[self.current_map]["accent"]
        now = time.time()

        if f == "arctic":
            # White-fog vignette at edges (blizzard)
            fog = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for i in range(30):
                a = int(90 * (1 - i / 30))
                m = i * min(WINDOW_W, WINDOW_H) // 60
                pygame.draw.rect(fog, (220, 230, 255, a),
                                 (m, m, WINDOW_W - 2*m, WINDOW_H - 2*m), 4)
            self.screen.blit(fog, (0, 0))
            # Pillar-box side bars (narrow horizontal FOV feel)
            bar_w = int(WINDOW_W * 0.07)
            bar = pygame.Surface((bar_w, WINDOW_H), pygame.SRCALPHA)
            bar.fill((180, 200, 240, 90))
            self.screen.blit(bar, (0, 0))
            self.screen.blit(bar, (WINDOW_W - bar_w, 0))
            # Drifting snow flecks over the camera feed
            self._update_and_draw_particles(dt, self.current_map)

        elif f == "warzone":
            # Orange smoke haze at edges
            haze = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for i in range(20):
                a = int(55 * (1 - i / 20))
                m = i * min(WINDOW_W, WINDOW_H) // 40
                pygame.draw.rect(haze, (40, 20, 5, a),
                                 (m, m, WINDOW_W - 2*m, WINDOW_H - 2*m), 6)
            self.screen.blit(haze, (0, 0))
            # Subtle animated dust streak
            self._update_and_draw_particles(dt, self.current_map)

        elif f == "jungle":
            # Green foliage vignette — lush corner darkening
            fog = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for i in range(25):
                a = int(70 * (1 - i / 25))
                m = i * min(WINDOW_W, WINDOW_H) // 50
                pygame.draw.rect(fog, (5, 30, 8, a),
                                 (m, m, WINDOW_W - 2*m, WINDOW_H - 2*m), 5)
            self.screen.blit(fog, (0, 0))
            # Rain streaks
            self._update_and_draw_particles(dt, self.current_map)

        elif f == "nightops":
            # Night-vision goggle scope mask (black outside oval)
            self.screen.blit(self._nv_mask, (0, 0))
            # Green CRT scanlines over the NV view
            nv_scan = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for y in range(0, WINDOW_H, 3):
                pygame.draw.line(nv_scan, (0, 60, 0, 18), (0, y), (WINDOW_W, y))
            self.screen.blit(nv_scan, (0, 0))
            # Phosphor glow ring around scope edge
            ring = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            rx, ry_ = int(WINDOW_W * 0.38), int(WINDOW_H * 0.48)
            cx, cy_ = WINDOW_W // 2, WINDOW_H // 2
            for t in range(3):
                pygame.draw.ellipse(ring, (0, 180, 60, 30 - t*8),
                                    (cx - rx - t*3, cy_ - ry_ - t*3,
                                     (rx + t*3)*2, (ry_ + t*3)*2), 2)
            self.screen.blit(ring, (0, 0))

        else:  # cyberpunk
            # Glitch horizontal bands (random, brief)
            if random.random() < 0.04:
                glitch = pygame.Surface((WINDOW_W, random.randint(2, 8)), pygame.SRCALPHA)
                glitch.fill((0, 210, 255, 40))
                self.screen.blit(glitch, (0, random.randint(0, WINDOW_H)))
            # Cyan edge fringe
            fringe = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            for i in range(15):
                a = int(50 * (1 - i / 15))
                m = i * min(WINDOW_W, WINDOW_H) // 30
                pygame.draw.rect(fringe, (0, 40, 60, a),
                                 (m, m, WINDOW_W - 2*m, WINDOW_H - 2*m), 4)
            self.screen.blit(fringe, (0, 0))

    # ── serial ───────────────────────────────────────────────

    def _find_pico_port(self):
        """Scan serial ports and return the first one that looks like a Pico."""
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            # First pass: explicit Pico/MicroPython keywords
            pico_keywords = ("pico", "micropython", "rp2", "rp2040", "rp2350")
            for p in ports:
                desc = (p.description or "").lower()
                mfr  = (p.manufacturer or "").lower()
                if any(k in desc or k in mfr for k in pico_keywords):
                    return p.device
            # Second pass: USB Serial Device (not Bluetooth)
            for p in ports:
                desc = (p.description or "").lower()
                if "usb serial" in desc and "bluetooth" not in desc:
                    return p.device
            # Last resort: any non-Bluetooth port
            for p in ports:
                desc = (p.description or "").lower()
                if "bluetooth" not in desc:
                    return p.device
        except Exception:
            pass
        return None

    def _init_serial(self):
        port = SERIAL_PORT
        if port == "AUTO":
            port = self._find_pico_port()
            if port:
                print(f"[OK] Auto-detected Pico on {port}")
            else:
                print("[!] AUTO port detection failed — no serial ports found")
                print("    Plug in the Pico and restart, or set SERIAL_PORT manually")
                return
        try:
            self._serial_conn = serial.Serial(port, SERIAL_BAUD, timeout=0.01)
            t = threading.Thread(target=self._serial_thread, daemon=True)
            t.start()
            print(f"[OK] Serial connected on {port} @ {SERIAL_BAUD} baud")
        except Exception as e:
            print(f"[!] Serial open failed ({port}): {e}")
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

        print(f"\n[*] Scanning {len(STREAM_CANDIDATES)} stream URLs for {PHONE_IP} …", flush=True)
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
        print("\n    Game will keep retrying — start IP Webcam on your phone and it will connect.")
        raise SystemExit  # caught by _connect_stream_bg, game keeps running

    def _connect_stream_bg(self):
        """Connect to camera in background — swaps into _capture when ready."""
        try:
            cap = self._connect_stream()
            self._capture.reconnect_cap(cap)
        except SystemExit:
            print("[!] Could not connect to any stream. Game will show black until camera is available.")
        finally:
            self._stream_connecting = False

    def _try_reconnect(self):
        """Attempt to reconnect to the last working URL via CameraCapture."""
        if self._capture.reconnect(self._active_url):
            print(f"[OK] Stream reconnected: {self._active_url}")

    # ── NPC tracking ─────────────────────────────────────────

    def _update_npc_tracking(self, new_boxes: list):
        """Match incoming detection boxes to existing NPC objects."""
        NPC_TIMEOUT = 0.4   # drop live NPC if not seen for this many seconds

        now = time.time()

        # Respawn dead NPCs whose cooldown has expired
        for npc in self.npcs:
            if npc.dead_until is not None and now >= npc.dead_until:
                npc.health         = npc.MAX_HEALTH
                npc.display_health = float(npc.MAX_HEALTH)
                npc.dead_until     = None
                npc.kill_icon_until = None
                npc.engaged        = False

        # Drop live NPCs not seen recently; keep dead NPCs until cooldown expires
        self.npcs = [n for n in self.npcs
                     if n.dead_until is not None or (now - n.last_seen) < NPC_TIMEOUT]

        def iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            ua = (a[2]-a[0])*(a[3]-a[1])
            ub = (b[2]-b[0])*(b[3]-b[1])
            return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0

        def size_ratio(a, b):
            wa = max(1, a[2] - a[0])
            wb = max(1, b[2] - b[0])
            return max(wa, wb) / min(wa, wb)

        # ── Greedy IoU-priority assignment ─────────────────────────────────
        # Include ALL NPCs (live + dead) so dead ones keep their tracked position
        all_indices = list(range(len(self.npcs)))
        det_boxes   = list(new_boxes)

        scores = []
        for di, (x1, y1, x2, y2) in enumerate(det_boxes):
            new_box = (x1, y1, x2, y2)
            for ni in all_indices:
                npc  = self.npcs[ni]
                nbox = tuple(int(v) for v in npc.display_box)
                ov   = iou(new_box, nbox)
                if ov > 0.15 and size_ratio(new_box, nbox) < 2.5:
                    scores.append((ov, di, ni))

        scores.sort(reverse=True)
        matched_det = set()
        matched_npc = set()
        for ov, di, ni in scores:
            if di in matched_det or ni in matched_npc:
                continue
            x1, y1, x2, y2 = det_boxes[di]
            self.npcs[ni].box       = (x1, y1, x2, y2)
            self.npcs[ni].last_seen = now
            matched_det.add(di)
            matched_npc.add(ni)

        # ── Tight distance fallback (only for live NPCs) ────────────────────
        live_indices = [i for i in all_indices if self.npcs[i].dead_until is None]
        for di, (x1, y1, x2, y2) in enumerate(det_boxes):
            if di in matched_det:
                continue
            new_box = (x1, y1, x2, y2)
            dcx = (x1 + x2) // 2
            dcy = (y1 + y2) // 2
            dw  = x2 - x1
            best_idx, best_d = None, float('inf')
            for ni in live_indices:
                if ni in matched_npc:
                    continue
                npc = self.npcs[ni]
                nx1, ny1, nx2, ny2 = npc.box
                d = math.hypot(dcx - (nx1+nx2)//2, dcy - (ny1+ny2)//2)
                if d < dw and d < best_d and size_ratio(new_box, (nx1,ny1,nx2,ny2)) < 2.0:
                    best_d, best_idx = d, ni
            if best_idx is not None:
                self.npcs[best_idx].box       = (x1, y1, x2, y2)
                self.npcs[best_idx].last_seen = now
                matched_det.add(di)
                matched_npc.add(best_idx)

        # ── Spawn new NPC for unmatched detections ──────────────────────────
        for di, (x1, y1, x2, y2) in enumerate(det_boxes):
            if di in matched_det:
                continue
            new_box = (x1, y1, x2, y2)
            # Block spawn if this detection overlaps any existing NPC (live or dead)
            overlaps = any(
                iou(new_box, tuple(int(v) for v in n.display_box)) > 0.1
                for n in self.npcs
            )
            if not overlaps and len(self.npcs) < 8:
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
        now = time.time()
        for npc in self.npcs:
            if npc.health <= 0 or (npc.dead_until is not None and npc.dead_until > now):
                continue
            bx1, by1, bx2, by2 = (int(v) for v in npc.display_box)
            x1, y1, x2, y2 = self._shrink_box(
                (bx1, by1, bx2, by2), HIT_SHRINK_X, HIT_SHRINK_Y)
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
        self.gun_recoil = 1.0
        self.shake_x    = random.uniform(-6, 6)
        self.shake_y    = random.uniform(-14, -6)
        self.shake_time = 0.10
        # Check if crosshair is on the exit button
        ex, ey, ew, eh = self._exit_btn_rect()
        if ex <= int(self.ch_x) <= ex + ew and ey <= int(self.ch_y) <= ey + eh:
            pygame.quit()
            sys.exit(0)

        # Check if crosshair is on a live NPC
        targeted = self._get_targeted_npc()
        if targeted:
            # ── Confirmed NPC hit — real gunshot + impact tink overlay ──
            self._play(self.snd_shoot)
            self._play(self.snd_hit_tink)
            damage = 25
            targeted.health    = max(0, targeted.health - damage)
            targeted.hit_flash = 0.25
            targeted.engaged   = True
            self.hit_markers.append(
                HitMarker(int(self.ch_x), int(self.ch_y), confirmed=True))
            self.score += 50

            if targeted.health <= 0:
                now = time.time()
                targeted.dead_until      = now + 6.0   # 6 s respawn cooldown
                targeted.kill_icon_until = now + 1.5   # skull visible for 1.5 s
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
        if self.reloading:
            return
        if self.ammo >= self.max_ammo:
            return   # already full — no sound, no animation
        self.reloading    = True
        self.reload_start = time.time()
        self.gun_reload_y = 0.0
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
        # ── Camera disconnect detection (phone streaming only) ──────────────
        if (self.state == "PLAYING" and not USE_WEBCAM_FALLBACK
                and not self._stream_connecting):
            stale = time.time() - self._capture.last_frame_time
            if stale > 5.0:
                print("[!] Camera stream lost — returning to connect screen")
                self.state = "CONNECTING"
                self._title_timer = 0.0
                self._stream_connecting = True
                threading.Thread(target=self._connect_stream_bg, daemon=True).start()
                return

        # Crosshair spread decay
        self.crosshair_spread = max(0, self.crosshair_spread - 30 * dt)

        # Flash timers
        self.damage_flash = max(0, self.damage_flash - dt)
        self.shoot_flash  = max(0, self.shoot_flash  - dt)

        # Re-zero flash timer
        if self._rezero_flash > 0:
            self._rezero_flash = max(0.0, self._rezero_flash - dt)

        # Screen shake
        if self.shake_time > 0:
            self.shake_time = max(0.0, self.shake_time - dt)
            frac = self.shake_time / 0.10
            self.shake_x = random.uniform(-6, 6) * frac
            self.shake_y = random.uniform(-14, -6) * frac
        else:
            self.shake_x = 0.0
            self.shake_y = 0.0

        # Gun recoil springs back
        self.gun_recoil = max(0.0, self.gun_recoil - dt * 9.0)

        # Idle gun bob
        self.gun_bob += dt * 1.8

        # Gun reload drop / return
        if self.reloading:
            progress = (time.time() - self.reload_start) / self.reload_time
            if progress < 0.35:
                self.gun_reload_y = progress / 0.35          # drop down
            else:
                self.gun_reload_y = max(0.0, 1.0 - (progress - 0.35) / 0.65)  # come back
        else:
            self.gun_reload_y = 0.0

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
                self._dbg_yaw, self._dbg_pitch = yaw, pitch
                if AIM_YAW_INVERT:   yaw   = -yaw
                if AIM_PITCH_INVERT: pitch = -pitch
                yaw   -= self._aim_yaw_offset
                pitch -= self._aim_pitch_offset
                # Map yaw  ±AIM_YAW_RANGE   → screen X 0..WINDOW_W
                #     pitch ±AIM_PITCH_RANGE → screen Y 0..WINDOW_H
                # Negative pitch = gun aimed up = crosshair near top (low Y)
                target_x = ( yaw   / AIM_YAW_RANGE   + 1.0) * 0.5 * WINDOW_W
                target_y = (-pitch / AIM_PITCH_RANGE  + 1.0) * 0.5 * WINDOW_H
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

        # ── map-specific dramatic visual filter ──────────────────────────
        # Apply contrast boost then the map's signature colour grade
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=-15)   # contrast base
        frame = self.apply_map_filter(frame)

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

    # Per-map health bar colour palettes — (high_hp, mid_hp, low_hp, targeted, outline)
    _MAP_HP_COLOURS = [
        # Arctic  — cyan/ice on cold blue bg, hot colours pop against white
        ((0, 220, 255),   (255, 210, 60),  (255, 60,  60),  (255, 80,  80),  (0, 60, 100)),
        # Warzone — bright orange/yellow on sepia, pops against warm tones
        ((255, 180, 30),  (255, 230, 0),   (220, 40,  40),  (255, 255, 80),  (80, 20, 0)),
        # Jungle  — white/yellow on green-tinted feed (green would vanish)
        ((240, 240, 60),  (255, 160, 30),  (220, 40,  40),  (255, 255, 180), (20, 50, 10)),
        # Night Ops — bright white/yellow on NV-green (green bar would vanish)
        ((220, 255, 180), (255, 230, 60),  (255, 80,  80),  (255, 255, 255), (0,  40, 20)),
        # Cyberpunk — neon cyan / magenta on dark hyper-saturated feed
        ((0,  230, 255),  (255, 0,   200), (255, 40,  80),  (255, 255, 0),   (20, 0,  50)),
    ]

    def _draw_skull(self, cx: int, cy: int, scale: float = 1.0):
        """Draw a skull-and-crossbones icon centred at (cx, cy) using pygame primitives."""
        s = scale

        def p(x, y):  # scale helper
            return (int(cx + x * s), int(cy + y * s))

        def r(x, y, w, h):  # scaled rect tuple
            return pygame.Rect(int(cx + x*s), int(cy + y*s), int(w*s), int(h*s))

        # ── Shadow pass (draw everything offset +2, dark) ─────────────────
        off = 2
        for col, ox, oy in [((0,0,0), off, off), ((255,255,255), 0, 0)]:
            # Crossbones — two diagonal bone lines
            bone_w = int(4 * s)
            # top-left → bottom-right bone
            pygame.draw.line(self.screen, col,
                             p(-14 + ox, -18 + oy), p(14 + ox, 6 + oy), bone_w)
            # top-right → bottom-left bone
            pygame.draw.line(self.screen, col,
                             p(14 + ox, -18 + oy), p(-14 + ox, 6 + oy), bone_w)
            # Knobby ends (circles at each bone tip)
            for bx, by in [(-14,-18),(14,-18),(-14,6),(14,6)]:
                pygame.draw.circle(self.screen, col,
                                   p(bx + ox, by + oy), int(5 * s))

            # Skull cranium
            pygame.draw.ellipse(self.screen, col, r(-11+ox, -20+oy, 22, 20))
            # Jaw
            pygame.draw.rect(self.screen, col, r(-8+ox, -4+oy, 16, 7), border_radius=int(3*s))

            # Eye sockets (punched out in black regardless of pass)
            eye_r = int(3.5 * s)
            pygame.draw.circle(self.screen, (0, 0, 0), p(-4+ox, -13+oy), eye_r)
            pygame.draw.circle(self.screen, (0, 0, 0), p(4+ox, -13+oy), eye_r)

            # Nose hole
            pygame.draw.rect(self.screen, (0, 0, 0), r(-1+ox, -8+oy, 2, 3))

            # Teeth — three small rects
            for tx in [-5, 0, 5]:
                pygame.draw.rect(self.screen, (0, 0, 0), r(tx+ox, -1+oy, 3, 4))

    def draw_npc_overlays(self):
        """Draw health bars for engaged/targeted NPCs, and death icons for killed ones."""
        m = min(self.current_map, len(self._MAP_HP_COLOURS) - 1)
        col_hi, col_mid, col_lo, col_targeted, col_outline = self._MAP_HP_COLOURS[m]

        targeted = self._get_targeted_npc()

        # ── Live NPC health bars (only if engaged or currently targeted) ──
        for npc in self.npcs:
            if npc.health <= 0:
                continue
            if not npc.engaged and npc is not targeted:
                continue  # don't clutter screen with bars for untouched NPCs

            x1, y1, x2, _ = (int(v) for v in npc.display_box)
            w = x2 - x1
            if w <= 0:
                continue

            pct   = npc.display_health / npc.MAX_HEALTH
            bar_h = 8
            bar_y = y1 - bar_h - 6

            if npc is targeted:
                bar_col = col_targeted
            elif npc.hit_flash > 0:
                bar_col = (255, 255, 255)
            elif pct > 0.5:
                bar_col = col_hi
            elif pct > 0.25:
                bar_col = col_mid
            else:
                bar_col = col_lo

            # Outline so bar shows on any background
            pygame.draw.rect(self.screen, col_outline, (x1 - 1, bar_y - 1, w + 2, bar_h + 2))
            pygame.draw.rect(self.screen, (10, 10, 10), (x1, bar_y, w, bar_h))
            pygame.draw.rect(self.screen, bar_col,
                             (x1, bar_y, max(1, int(w * pct)), bar_h))

        # ── Death icons + respawn countdown — drawn at the NPC's tracked position ──
        now = time.time()
        for npc in self.npcs:
            if npc.dead_until is None or npc.dead_until <= now:
                continue
            if npc.kill_icon_until is None or now > npc.kill_icon_until:
                continue   # icon window expired — cooldown still active, just don't draw
            bx1, by1, bx2, by2 = (int(v) for v in npc.display_box)
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            remaining = npc.dead_until - now
            total_cd  = 6.0

            self._draw_skull(cx, cy)

            # Circular respawn cooldown ring around the skull
            ring_r = 36
            prog = remaining / total_cd   # 1.0 → 0.0
            arc_deg = int(360 * prog)
            if arc_deg > 0:
                rect = pygame.Rect(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
                pygame.draw.arc(self.screen, (220, 30, 30), rect,
                                math.radians(90),
                                math.radians(90 + arc_deg), 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), ring_r, 1)

            cd_txt = self.font_small.render(f"{remaining:.1f}s", True, (220, 30, 30))
            self.screen.blit(cd_txt, cd_txt.get_rect(center=(cx, cy + ring_r + 12)))

    def draw_health_bar(self):
        """Valorant-style: large colour-coded HP number + thin bar."""
        x, y = 40, WINDOW_H - 70
        pct = self.health / 100
        m = min(self.current_map, len(self._MAP_HP_COLOURS) - 1)
        col_hi, col_mid, col_lo = self._MAP_HP_COLOURS[m][:3]
        col = col_hi if pct > 0.5 else col_mid if pct > 0.25 else col_lo

        # Dark panel background
        panel = pygame.Surface((210, 68), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 130))
        self.screen.blit(panel, (x - 8, y - 58))
        pygame.draw.line(self.screen, col, (x - 8, y - 58), (x + 202, y - 58), 1)

        # Large HP number
        big = self.font_big.render(str(self.health), True, col)
        self.screen.blit(big, (x, y - big.get_height()))

        # Thin bar below
        bw, bh = 180, 4
        by = y + 4
        pygame.draw.rect(self.screen, (40, 40, 40), (x, by, bw, bh))
        pygame.draw.rect(self.screen, col, (x, by, max(0, int(bw * pct)), bh))

        # Small label
        lbl = self.font_small.render("HP", True, (80, 80, 80))
        self.screen.blit(lbl, (x + big.get_width() + 8, y - lbl.get_height()))

    def draw_ammo(self):
        """CS:GO-style: large ammo number / reserve + reload bar."""
        # Dark panel background (bottom-right)
        panel = pygame.Surface((190, 68), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 130))
        self.screen.blit(panel, (WINDOW_W - 198, WINDOW_H - 76))
        m = min(self.current_map, len(self._MAP_HP_COLOURS) - 1)
        acc = self._MAP_HP_COLOURS[m][0]
        pygame.draw.line(self.screen, acc,
                         (WINDOW_W - 198, WINDOW_H - 76), (WINDOW_W - 8, WINDOW_H - 76), 1)
        if self.reloading:
            progress = min(1.0, (time.time() - self.reload_start) / self.reload_time)
            # RELOADING label
            txt = self.font_med.render("RELOADING", True, ORANGE)
            rx = WINDOW_W - txt.get_width() - 40
            ry = WINDOW_H - 100
            self.screen.blit(txt, (rx, ry))
            # Progress bar
            bw, bh = txt.get_width(), 4
            by = ry + txt.get_height() + 4
            pygame.draw.rect(self.screen, (40, 40, 40), (rx, by, bw, bh))
            pygame.draw.rect(self.screen, ORANGE, (rx, by, int(bw * progress), bh))
        else:
            # Large "6 / 30" counter bottom-right
            col_cur = WHITE if self.ammo > self.max_ammo // 3 else (220, 80, 80)
            cur_txt = self.font_big.render(str(self.ammo), True, col_cur)
            sep_txt = self.font_hud.render(" /", True, (80, 80, 80))
            res_txt = self.font_hud.render(f" {self.max_ammo}", True, (80, 80, 80))

            total_w = cur_txt.get_width() + sep_txt.get_width() + res_txt.get_width()
            rx = WINDOW_W - total_w - 40
            ry = WINDOW_H - cur_txt.get_height() - 20

            self.screen.blit(cur_txt, (rx, ry))
            self.screen.blit(sep_txt, (rx + cur_txt.get_width(), ry + cur_txt.get_height() - sep_txt.get_height()))
            self.screen.blit(res_txt, (rx + cur_txt.get_width() + sep_txt.get_width(), ry + cur_txt.get_height() - res_txt.get_height()))

            lbl = self.font_small.render("AMMO", True, (60, 60, 60))
            self.screen.blit(lbl, (WINDOW_W - lbl.get_width() - 40, ry - lbl.get_height() - 2))

    def draw_muzzle_flash(self):
        """Orange/white burst near crosshair when firing."""
        if self.shoot_flash <= 0.04:
            return
        frac = self.shoot_flash / 0.08
        tip_x = int(self.ch_x) + int(self.gun_recoil * -10)
        tip_y = int(self.ch_y) + int(self.gun_recoil * -10)

        size = 120
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        r = int(28 * frac)
        pygame.draw.circle(surf, (255, 255, 220, int(220 * frac)), (cx, cy), r)
        pygame.draw.circle(surf, (255, 145, 20,  int(180 * frac)), (cx, cy), int(r * 1.6))
        for i in range(8):
            a = math.radians(i * 45)
            ex = cx + int(math.cos(a) * int(52 * frac))
            ey = cy + int(math.sin(a) * int(52 * frac))
            pygame.draw.line(surf, (255, 185, 40, int(130 * frac)), (cx, cy), (ex, ey), 2)
        self.screen.blit(surf, (tip_x - cx, tip_y - cy))

    def draw_volume_slider(self):
        """Always-visible volume slider in the top-right corner."""
        sx, sy = WINDOW_W - 210, 12   # track left edge, vertical centre
        sw, sh = 160, 6               # track width, height
        vol = self._current_volume
        knob_x = sx + int(sw * vol)

        # "VOL" label
        lbl = self.font_small.render("VOL", True, HUD_DIM)
        self.screen.blit(lbl, (sx - lbl.get_width() - 8, sy - lbl.get_height() // 2))

        # Track background
        pygame.draw.rect(self.screen, (35, 35, 35), (sx, sy - sh // 2, sw, sh), border_radius=3)
        # Filled portion
        if int(sw * vol) > 0:
            pygame.draw.rect(self.screen, HUD_GREEN,
                (sx, sy - sh // 2, int(sw * vol), sh), border_radius=3)
        # Track outline
        pygame.draw.rect(self.screen, (60, 60, 60), (sx, sy - sh // 2, sw, sh), 1, border_radius=3)

        # Knob circle
        knob_col = WHITE if self._vol_dragging else (200, 200, 200)
        pygame.draw.circle(self.screen, knob_col, (knob_x, sy), 8)
        pygame.draw.circle(self.screen, (30, 30, 30), (knob_x, sy), 8, 1)

        # Percentage label
        pct = self.font_small.render(f"{int(vol * 100)}%", True, HUD_DIM)
        self.screen.blit(pct, (sx + sw + 6, sy - pct.get_height() // 2))

    def _toggle_fullscreen(self):
        global WINDOW_W, WINDOW_H
        flags = self.screen.get_flags()
        if flags & pygame.FULLSCREEN:
            WINDOW_W, WINDOW_H = 1280, 720
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        else:
            info = pygame.display.Info()
            WINDOW_W = info.current_w
            WINDOW_H = info.current_h
            self.screen = pygame.display.set_mode(
                (WINDOW_W, WINDOW_H), pygame.FULLSCREEN)
        # Regenerate resolution-dependent overlays
        self._vignette     = self._make_vignette()
        self._scanlines    = self._make_scanlines()
        self._grain_frames = self._make_grain_frames()
        self._nv_mask      = self._make_nv_mask()
        self._particles.clear()
        # Re-centre crosshair
        self.ch_x = float(WINDOW_W // 2)
        self.ch_y = float(WINDOW_H // 2)

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
        self.screen.blit(txt, (WINDOW_W - 80, 28))

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

    def _exit_btn_rect(self):
        """Returns (x, y, w, h) of the exit button hit area."""
        return (14, 14, 70, 26)

    def draw_exit_button(self):
        x, y, w, h = self._exit_btn_rect()
        cx, cy = int(self.ch_x), int(self.ch_y)
        hovered = x <= cx <= x + w and y <= cy <= y + h
        col = (220, 50, 50) if hovered else (120, 30, 30)
        pygame.draw.rect(self.screen, col, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, (200, 80, 80) if hovered else (80, 20, 20),
                         (x, y, w, h), 1, border_radius=4)
        lbl = self.font_small.render("[ EXIT ]", True, (255, 200, 200) if hovered else (180, 80, 80))
        self.screen.blit(lbl, (x + (w - lbl.get_width()) // 2, y + (h - lbl.get_height()) // 2))

    def draw_hud_title(self):
        txt = self.font_small.render("ROBO-HUNTER  //  HACKABOT 2026", True, (50, 100, 60))
        self.screen.blit(txt, (100, 14))

    def draw_serial_status(self):
        if ENABLE_SERIAL:
            status = "[PICO CONNECTED]" if self._serial_conn else "[PICO OFFLINE]"
            col    = HUD_DIM if self._serial_conn else (100, 40, 40)
            txt    = self.font_small.render(status, True, col)
            self.screen.blit(txt, (WINDOW_W - txt.get_width() - 10, WINDOW_H - 24))
            if self._serial_conn:
                dbg = self.font_small.render(
                    f"YAW:{self._dbg_yaw:>6.1f}  PITCH:{self._dbg_pitch:>6.1f}", True, HUD_DIM)
                self.screen.blit(dbg, (WINDOW_W - dbg.get_width() - 10, WINDOW_H - 44))

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

    # ── music ─────────────────────────────────────────────────

    def _set_music(self, track: str):
        """Switch background music. track = 'menu' | 'game' | None"""
        if track == self._music_track:
            return
        self._music_track = track
        try:
            if track == "menu":
                path = os.path.join(_ASSETS_DIR, "menu_music.mp3")
                if os.path.exists(path):
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.set_volume(self._current_volume * MUSIC_VOLUME)
                    pygame.mixer.music.play(-1)
                else:
                    print(f"[!] menu_music.mp3 not found at {path}")
            elif track == "game":
                path = os.path.join(_ASSETS_DIR, "game_start_music.mp3")
                if os.path.exists(path):
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.set_volume(self._current_volume * MUSIC_VOLUME)
                    pygame.mixer.music.play(-1)
                else:
                    print(f"[!] game_start_music.mp3 not found at {path}")
            elif track is None:
                pygame.mixer.music.stop()
        except Exception as e:
            print(f"[!] Music switch failed ({track}): {e}")

    # ── start new game ───────────────────────────────────────

    def _start_new_game(self):
        self.score          = 0
        self.health         = 100
        self.ammo           = self.max_ammo
        self.kills          = 0
        self.game_active    = True
        self.shoot_cooldown = 0.15

    # ── map video filter ─────────────────────────────────────

    def apply_map_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply a dramatic map-specific visual filter to the live BGR frame."""
        f = MAP_DATA[self.current_map]["filter"]

        if f == "arctic":
            # Heavy desaturation + cold blue shift
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(frame, 0.15, grey3, 0.85, 0)
            # Blue channel boost, red drop
            out = out.astype(np.int16)
            out[:, :, 0] = np.clip(out[:, :, 0] + 55, 0, 255)  # blue
            out[:, :, 2] = np.clip(out[:, :, 2] - 25, 0, 255)  # red
            return out.astype(np.uint8)

        elif f == "warzone":
            # Harsh sepia + warm orange grade + contrast crunch
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eq   = cv2.equalizeHist(grey)
            warm = cv2.merge([
                np.clip(eq.astype(np.int32) + 10, 0, 255).astype(np.uint8),   # blue-ish
                np.clip(eq.astype(np.int32) - 5,  0, 255).astype(np.uint8),   # green
                np.clip(eq.astype(np.int32) + 35, 0, 255).astype(np.uint8),   # red
            ])
            return cv2.addWeighted(warm, 0.85, frame, 0.15, 0)

        elif f == "jungle":
            # Aggressive green channel push + slight haze
            out = frame.astype(np.int16).copy()
            out[:, :, 1] = np.clip(out[:, :, 1] + 55, 0, 255)  # green
            out[:, :, 0] = np.clip(out[:, :, 0] - 20, 0, 255)  # blue
            out[:, :, 2] = np.clip(out[:, :, 2] - 30, 0, 255)  # red
            # Slight blur for atmospheric haze
            blurred = cv2.GaussianBlur(out.astype(np.uint8), (3, 3), 0)
            return cv2.addWeighted(out.astype(np.uint8), 0.7, blurred, 0.3, 0)

        elif f == "nightops":
            # True night-vision green — CLAHE contrast + green channel only
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            eq = clahe.apply(grey)
            # Add mild phosphor glow (slight blur overlay)
            glow = cv2.GaussianBlur(eq, (5, 5), 0)
            eq = np.clip(eq.astype(np.int16) + glow.astype(np.int16) // 4, 0, 255).astype(np.uint8)
            nv = cv2.merge([
                np.zeros_like(eq),
                np.clip(eq.astype(np.int32) - 20, 0, 255).astype(np.uint8),
                np.zeros_like(eq),
            ])
            return nv

        else:  # cyberpunk
            # Max saturation + chromatic aberration shift + neon tint
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.2, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.80, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            # Chromatic aberration: shift red channel right by 3px
            r_shifted = np.roll(out[:, :, 2], 3, axis=1)
            out[:, :, 2] = r_shifted
            # Neon overlay
            out = out.astype(np.int16)
            out[:, :, 0] = np.clip(out[:, :, 0] + 30, 0, 255)  # blue
            out[:, :, 2] = np.clip(out[:, :, 2] - 10, 0, 255)  # reduce red
            return out.astype(np.uint8)

    # ── menu background ──────────────────────────────────────

    def _draw_menu_bg(self, map_idx: int = -1, dt: float = 0.0):
        """Dark FPS lobby background with animated atmosphere."""
        bg = MAP_DATA[map_idx]["bg"] if map_idx >= 0 else (6, 8, 14)
        self.screen.fill(bg)

        accent = MAP_DATA[map_idx]["accent"] if map_idx >= 0 else (0, 180, 100)

        # Animated diagonal slash lines (scrolling slowly)
        ga = (int(accent[0]*0.06), int(accent[1]*0.06), int(accent[2]*0.06))
        scroll = int(self._title_timer * 18) % 90
        for d in range(-WINDOW_H, WINDOW_W + WINDOW_H, 90):
            x1 = d + scroll
            pygame.draw.line(self.screen, ga, (x1, 0), (x1 + WINDOW_H, WINDOW_H))

        # Film grain
        self._draw_grain(dt)

        # Ambient particles
        if map_idx >= 0:
            self._update_and_draw_particles(dt, map_idx)

        # Top bar
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, WINDOW_W, 64))
        pygame.draw.line(self.screen, accent, (0, 64), (WINDOW_W, 64), 1)

        # Game title left-aligned in top bar
        t1 = self.font_big.render("ROBO", True, (255, 255, 255))
        t2 = self.font_big.render("-HUNTER", True, accent)
        self.screen.blit(t1, (30, 10))
        self.screen.blit(t2, (30 + t1.get_width(), 10))

        # "HACKABOT 2026" right-aligned
        badge = self.font_small.render("HACKABOT  2026", True, (80, 90, 80))
        self.screen.blit(badge, (WINDOW_W - badge.get_width() - 30, 24))

        # Bottom bar
        pygame.draw.rect(self.screen, (0, 0, 0), (0, WINDOW_H - 40, WINDOW_W, 40))
        pygame.draw.line(self.screen, accent, (0, WINDOW_H - 40), (WINDOW_W, WINDOW_H - 40), 1)
        hint = self.font_small.render(
            "↑↓  NAVIGATE    SPACE / ENTER  SELECT    ESC  QUIT    C  ZERO AIM    F11  FULLSCREEN",
            True, (60, 70, 60))
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 20)))

    # ── connecting screen ─────────────────────────────────────

    def _draw_connecting_screen(self, dt: float):
        """Blocks at launch until the phone camera is live."""
        self._title_timer += dt
        self._draw_menu_bg(dt=dt)

        # Spinner dots
        dots = "." * (int(self._title_timer * 2) % 4)

        msg = self.font_big.render("CONNECTING TO CAMERA", True, (200, 200, 200))
        self.screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 70)))

        sub = self.font_hud.render(f"Waiting{dots}", True, (0, 180, 100))
        self.screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2)))

        hint = self.font_small.render(
            "Start  IP Webcam  on your phone  then  tap  'Start server'",
            True, (50, 70, 50))
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 60)))

        # Animated scanning line
        sweep_y = int((self._title_timer * 100) % WINDOW_H)
        sl = pygame.Surface((WINDOW_W, 2), pygame.SRCALPHA)
        sl.fill((0, 180, 100, 20))
        self.screen.blit(sl, (0, sweep_y))

        # Once connected, auto-advance to title
        if not self._stream_connecting:
            self.state = "TITLE"
            self._title_timer = 0.0

    # ── title screen ─────────────────────────────────────────

    def _draw_title_screen(self, dt: float):
        self._title_timer += dt
        self._draw_menu_bg(dt=dt)

        # Animated scanline sweep
        sweep_y = int((self._title_timer * 120) % WINDOW_H)
        scan_surf = pygame.Surface((WINDOW_W, 2), pygame.SRCALPHA)
        scan_surf.fill((0, 255, 160, 18))
        self.screen.blit(scan_surf, (0, sweep_y))

        # Big centred title
        cy = WINDOW_H // 2 - 60
        shadow = self.font_big.render("ROBO-HUNTER", True, (0, 0, 0))
        title  = self.font_big.render("ROBO-HUNTER", True, (255, 255, 255))
        self.screen.blit(shadow, shadow.get_rect(center=(WINDOW_W // 2 + 3, cy + 3)))
        self.screen.blit(title,  title.get_rect(center=(WINDOW_W // 2, cy)))

        # Accent line under title
        lw = title.get_width() + 40
        lx = WINDOW_W // 2 - lw // 2
        pygame.draw.line(self.screen, (0, 255, 160), (lx, cy + 36), (lx + lw, cy + 36), 2)

        # Subtitle
        sub = self.font_med.render("HACKABOT  2026  LIVE  ARENA", True, (120, 130, 120))
        self.screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, cy + 60)))

        # Pulsing "PRESS SPACE" prompt
        pulse = 0.55 + 0.45 * math.sin(self._title_timer * 2.8)
        col = (int(0 * pulse), int(230 * pulse), int(140 * pulse))
        msg = self.font_hud.render("PRESS  SPACE  TO  DEPLOY", True, col)
        self.screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, cy + 130)))

        # Version tag bottom-left
        ver = self.font_small.render("v2.0  //  LIVE  CAM  MODE", True, (40, 50, 40))
        self.screen.blit(ver, (30, WINDOW_H - 60))

    # ── map select screen ────────────────────────────────────

    def _draw_map_select_screen(self, dt: float):
        self._title_timer += dt
        m = self.current_map
        self._draw_menu_bg(m, dt)

        content_top = 74
        content_bot = WINDOW_H - 40

        # ── LEFT PANEL: map list ─────────────────────────────────────────────
        panel_w = 340
        # Semi-transparent left panel
        panel = pygame.Surface((panel_w, content_bot - content_top), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        self.screen.blit(panel, (0, content_top))

        label = self.font_small.render("SELECT  MAP", True, (120, 130, 100))
        self.screen.blit(label, (24, content_top + 14))
        pygame.draw.line(self.screen, (40, 50, 40),
                         (24, content_top + 32), (panel_w - 24, content_top + 32), 1)

        item_h = 54
        list_y = content_top + 44
        for i, md in enumerate(MAP_DATA):
            y = list_y + i * item_h
            accent = md["accent"]
            selected = (i == m)

            if selected:
                # Highlight bar
                hl = pygame.Surface((panel_w - 4, item_h - 6), pygame.SRCALPHA)
                hl.fill((accent[0]//6, accent[1]//6, accent[2]//6, 200))
                self.screen.blit(hl, (2, y))
                # Left accent stripe
                pygame.draw.rect(self.screen, accent, (0, y, 4, item_h - 6))

            # Map index number
            num_col = accent if selected else (50, 55, 50)
            num = self.font_small.render(f"{i+1:02d}", True, num_col)
            self.screen.blit(num, (14, y + 10))

            # Map name
            name_col = (255, 255, 255) if selected else (130, 140, 130)
            name_font = self.font_med if selected else self.font_small
            name_surf = name_font.render(md["name"], True, name_col)
            self.screen.blit(name_surf, (50, y + (item_h // 2 - name_surf.get_height() // 2) - 3))

            # Tags on selected
            if selected:
                tx = 50
                for tag in md["tags"][:2]:
                    ts = self.font_small.render(tag, True, accent)
                    self.screen.blit(ts, (tx, y + item_h - 20))
                    tx += ts.get_width() + 12

        # ── RIGHT PANEL: map preview ─────────────────────────────────────────
        right_x = panel_w + 20
        right_w  = WINDOW_W - right_x - 20
        right_h  = content_bot - content_top - 10
        right_y  = content_top + 5

        md = MAP_DATA[m]
        accent = md["accent"]
        bg_col = md["bg"]

        # Panel background — gradient
        for i in range(right_h):
            alpha = i / right_h
            r = int(bg_col[0] * (1 - alpha * 0.6))
            g = int(bg_col[1] * (1 - alpha * 0.6))
            b = int(bg_col[2] * (1 - alpha * 0.6))
            pygame.draw.line(self.screen, (r, g, b),
                             (right_x, right_y + i), (right_x + right_w, right_y + i))

        # Terrain silhouette at the bottom of the right panel
        terrain_h = int(right_h * 0.42)
        self._draw_map_terrain(right_x, right_y + right_h - terrain_h,
                               right_w, terrain_h, md["filter"], accent)

        # Animated corner brackets
        t = self._title_timer
        blen = int(30 + 20 * abs(math.sin(t * 1.5)))
        bthk = 2
        corners = [
            (right_x + 8, right_y + 8),
            (right_x + right_w - 8, right_y + 8),
            (right_x + 8, right_y + right_h - 8),
            (right_x + right_w - 8, right_y + right_h - 8),
        ]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            pygame.draw.line(self.screen, accent, (cx, cy), (cx + blen * dx, cy), bthk)
            pygame.draw.line(self.screen, accent, (cx, cy), (cx, cy + blen * dy), bthk)

        # Map name — large
        name_shadow = self.font_big.render(md["name"], True, (0, 0, 0))
        name_surf   = self.font_big.render(md["name"], True, accent)
        nx = right_x + right_w // 2
        ny = right_y + right_h // 2 - 80
        self.screen.blit(name_shadow, name_shadow.get_rect(center=(nx + 3, ny + 3)))
        self.screen.blit(name_surf,   name_surf.get_rect(center=(nx, ny)))

        # Accent line under name
        nw = name_surf.get_width()
        pygame.draw.line(self.screen, accent,
                         (nx - nw // 2, ny + 30), (nx + nw // 2, ny + 30), 2)

        # Subtitle
        sub = self.font_med.render(md["sub"], True, (160, 165, 155))
        self.screen.blit(sub, sub.get_rect(center=(nx, ny + 55)))

        # Tags row
        tag_total_w = sum(
            self.font_small.size(f"  {tag}  ")[0] + 6 for tag in md["tags"]
        )
        tx = nx - tag_total_w // 2
        ty = ny + 90
        for tag in md["tags"]:
            tw, th = self.font_small.size(f"  {tag}  ")
            pygame.draw.rect(self.screen, (int(accent[0]*0.18), int(accent[1]*0.18), int(accent[2]*0.18)),
                             (tx, ty, tw + 6, th + 6), border_radius=3)
            pygame.draw.rect(self.screen, accent, (tx, ty, tw + 6, th + 6), 1, border_radius=3)
            ts = self.font_small.render(f"  {tag}  ", True, accent)
            self.screen.blit(ts, (tx + 3, ty + 3))
            tx += tw + 6 + 8

        # "DEPLOY" button — animated
        pulse = 0.6 + 0.4 * math.sin(self._title_timer * 3.0)
        btn_col  = (int(accent[0] * pulse), int(accent[1] * pulse), int(accent[2] * pulse))
        btn_surf = self.font_hud.render("[ SPACE ]  DEPLOY", True, btn_col)
        by = right_y + right_h - 60
        self.screen.blit(btn_surf, btn_surf.get_rect(center=(nx, by)))

        # Scanline sweep across right panel
        sweep_y = right_y + int((self._title_timer * 80) % right_h)
        sl = pygame.Surface((right_w, 2), pygame.SRCALPHA)
        sl.fill((accent[0], accent[1], accent[2], 14))
        self.screen.blit(sl, (right_x, sweep_y))

    # ── calibrating screen ───────────────────────────────────

    def _draw_calibrating_screen(self, dt: float):
        self._calib_timer += dt
        self._draw_menu_bg(self.current_map, dt)

        accent = MAP_DATA[self.current_map]["accent"]
        cx = WINDOW_W // 2
        cy = WINDOW_H // 2

        # Title
        title = self.font_big.render("CALIBRATING GUN", True, accent)
        self.screen.blit(title, title.get_rect(center=(cx, cy - 80)))

        # Instructions
        instr = self.font_med.render("Hold gun level and still...", True, (160, 165, 155))
        self.screen.blit(instr, instr.get_rect(center=(cx, cy - 30)))

        # Progress bar
        bar_w, bar_h = 400, 18
        bx = cx - bar_w // 2
        by = cy + 20
        elapsed = max(0.0, self._calib_timer - self._calib_delay)
        progress = min(1.0, elapsed / self._calib_duration)

        pygame.draw.rect(self.screen, (30, 35, 30), (bx, by, bar_w, bar_h), border_radius=4)
        if progress > 0:
            pygame.draw.rect(self.screen, accent,
                             (bx, by, int(bar_w * progress), bar_h), border_radius=4)
        pygame.draw.rect(self.screen, (80, 90, 80), (bx, by, bar_w, bar_h), 1, border_radius=4)

        # Percentage
        pct_txt = self.font_small.render(f"{int(progress * 100)}%", True, (160, 165, 155))
        self.screen.blit(pct_txt, pct_txt.get_rect(center=(cx, by + bar_h + 16)))

        # Done — advance to countdown
        if progress >= 1.0:
            self._set_music("game")
            self.state = "COUNTDOWN"
            self._countdown_val   = 3
            self._countdown_timer = 0.0

        # GET READY flash once bar fills
        if progress >= 0.98:
            ready = self.font_hud.render("GET READY", True, accent)
            self.screen.blit(ready, ready.get_rect(center=(cx, cy + 90)))

    # ── countdown screen ─────────────────────────────────────

    def _draw_countdown_screen(self, dt: float):
        self._countdown_timer += dt
        self._draw_menu_bg(self.current_map, dt)

        accent = MAP_DATA[self.current_map]["accent"]
        cx = WINDOW_W // 2
        cy = WINDOW_H // 2

        # Map name banner
        map_lbl = self.font_med.render(MAP_DATA[self.current_map]["name"], True, (160, 165, 155))
        self.screen.blit(map_lbl, map_lbl.get_rect(center=(cx, cy - 100)))

        # Countdown number — pulsing
        if self._countdown_timer >= 1.0:
            self._countdown_val -= 1
            self._countdown_timer = 0.0

        if self._countdown_val <= 0:
            self._start_new_game()
            self.state = "PLAYING"
            return

        pulse = 1.0 - (self._countdown_timer * 0.6)
        num_col = (int(accent[0] * pulse), int(accent[1] * pulse), int(accent[2] * pulse))
        num_surf = self.font_big.render(str(self._countdown_val), True, num_col)
        self.screen.blit(num_surf, num_surf.get_rect(center=(cx, cy)))

        deploy = self.font_med.render("DEPLOYING...", True, (120, 130, 120))
        self.screen.blit(deploy, deploy.get_rect(center=(cx, cy + 70)))

    # ── end / debrief screen ─────────────────────────────────

    def _draw_end_screen(self):
        m = self.current_map
        accent = MAP_DATA[m]["accent"]

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        self.screen.blit(overlay, (0, 0))

        # Top banner
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, WINDOW_W, 70))
        pygame.draw.line(self.screen, accent, (0, 70), (WINDOW_W, 70), 2)

        title = self.font_big.render("MISSION  DEBRIEF", True, accent)
        self.screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 35)))

        # Stats box
        box_w, box_h = 500, 260
        bx = WINDOW_W // 2 - box_w // 2
        by = 110
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 160))
        self.screen.blit(box, (bx, by))
        pygame.draw.rect(self.screen, accent, (bx, by, box_w, box_h), 1)
        pygame.draw.rect(self.screen, accent, (bx, by, box_w, 30))
        lbl = self.font_small.render("COMBAT  STATISTICS", True, (0, 0, 0))
        self.screen.blit(lbl, lbl.get_rect(center=(WINDOW_W // 2, by + 14)))

        stats = [
            ("SCORE",    str(self.score)),
            ("KILLS",    str(self.kills)),
            ("MAP",      MAP_DATA[m]["name"]),
        ]
        for j, (label, val) in enumerate(stats):
            y = by + 45 + j * 60
            pygame.draw.line(self.screen, (30, 35, 30),
                             (bx + 20, y + 44), (bx + box_w - 20, y + 44), 1)
            lsurf = self.font_small.render(label, True, (120, 130, 110))
            vsurf = self.font_hud.render(val, True, (240, 240, 240))
            self.screen.blit(lsurf, (bx + 24, y))
            self.screen.blit(vsurf, (bx + box_w - vsurf.get_width() - 24, y - 4))

        restart = self.font_hud.render("[ R ]  REDEPLOY", True, accent)
        self.screen.blit(restart, restart.get_rect(center=(WINDOW_W // 2, by + box_h + 50)))

        map_label = self.font_small.render(f"MAP:  {MAP_DATA[m]['name']}  //  {MAP_DATA[m]['sub']}", True, (70, 80, 70))
        self.screen.blit(map_label, map_label.get_rect(center=(WINDOW_W // 2, WINDOW_H - 30)))

    # ── main loop ────────────────────────────────────────────

    def run(self):
        running   = True
        prev_time = time.time()

        print("[*] Game running!")
        print("    SPACE / ENTER        = confirm / shoot")
        print("    UP / DOWN            = navigate menus")
        print("    R                    = reload / redeploy")
        print("    D                    = simulate taking damage")
        print("    ESC                  = return to menu / quit")
        print("    C                    = zero aim")
        print("    F11                  = toggle fullscreen")
        if ENABLE_SERIAL:
            print(f"    Pico 1 serial        = {SERIAL_PORT} @ {SERIAL_BAUD} baud")
        if ENABLE_DETECTION:
            method = "YOLOv8n" if (USE_YOLO and YOLO_AVAILABLE) else "HOG"
            print(f"    Detection method     = {method}")

        while running:
            now      = time.time()
            dt       = min(now - prev_time, 0.05)   # cap dt to avoid spiral on lag
            prev_time = now

            # ── EVENTS ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    # Global keys (always active)
                    if event.key == pygame.K_ESCAPE:
                        if self.state == "PLAYING":
                            # Return to map select, reset game state
                            self.state = "MAP_SELECT"
                            self.game_active = False
                            self.npcs.clear()
                            self._particles.clear()
                            self._set_music("menu")
                        else:
                            running = False
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    if event.key == pygame.K_c:
                        self._aim_yaw_offset   = self._dbg_yaw
                        self._aim_pitch_offset = self._dbg_pitch
                        self.ch_x = float(WINDOW_W // 2)
                        self.ch_y = float(WINDOW_H // 2)
                        self._rezero_flash = 1.5
                    if event.key == pygame.K_LEFTBRACKET:
                        self._set_volume(max(0.0, self._current_volume - 0.1))
                    if event.key == pygame.K_RIGHTBRACKET:
                        self._set_volume(min(1.0, self._current_volume + 0.1))

                    # ── Menu / state navigation ──
                    if self.state == "TITLE":
                        if event.key == pygame.K_SPACE:
                            self.state = "MAP_SELECT"
                    elif self.state == "MAP_SELECT":
                        if event.key == pygame.K_UP:
                            self.current_map = (self.current_map - 1) % len(MAP_DATA)
                        elif event.key == pygame.K_DOWN:
                            self.current_map = (self.current_map + 1) % len(MAP_DATA)
                        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            self.state = "CALIBRATING"
                            self._calib_timer = 0.0
                    elif self.state == "END":
                        if event.key == pygame.K_r:
                            self.state = "TITLE"
                            self._title_timer = 0.0
                            self._set_music("menu")

                    # ── In-game keys ──
                    if self.state == "PLAYING":
                        if event.key == pygame.K_r:
                            self.start_reload()
                        if event.key == pygame.K_SPACE:
                            self.shoot()
                        if event.key == pygame.K_d:
                            self.take_damage()

                # Mouse moves crosshair when gun serial is not connected (testing)
                if event.type == pygame.MOUSEMOTION:
                    if not ENABLE_SERIAL:
                        self.ch_x = float(event.pos[0])
                        self.ch_y = float(event.pos[1])
                    if self._vol_dragging:
                        sx = WINDOW_W - 210
                        sw = 160
                        frac = (event.pos[0] - sx) / sw
                        self._set_volume(max(0.0, min(1.0, frac)))

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self._vol_dragging = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # Check volume slider hit
                        sx, sy, sw, sh = WINDOW_W - 210, 12, 160, 14
                        if (sx <= event.pos[0] <= sx + sw and
                                sy - 10 <= event.pos[1] <= sy + sh + 10):
                            self._vol_dragging = True
                            frac = (event.pos[0] - sx) / sw
                            self._set_volume(max(0.0, min(1.0, frac)))
                        elif self.state == "PLAYING" and self.game_active:
                            self.shoot()

            # ── UPDATE ──────────────────────────────────────
            if self.state == "PLAYING":
                self.update(dt)
                # Transition to END when health depleted
                if not self.game_active:
                    self.state = "END"

            # ── DRAW ────────────────────────────────────────
            self.screen.fill((0, 0, 0))

            # ── Menu states — no camera feed needed ──
            if self.state == "CONNECTING":
                self._draw_connecting_screen(dt)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            if self.state == "TITLE":
                self._draw_title_screen(dt)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            if self.state == "MAP_SELECT":
                self._draw_map_select_screen(dt)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            if self.state == "CALIBRATING":
                self._draw_calibrating_screen(dt)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            if self.state == "COUNTDOWN":
                self._draw_countdown_screen(dt)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            # ── PLAYING / END — show live camera feed ──

            # Show connecting screen while stream isn't ready yet
            if self._stream_connecting:
                msg = self.font_med.render("Connecting to camera...", True, (80, 180, 80))
                sub = self.font_small.render("Start IP Webcam on your phone", True, (60, 100, 60))
                self.screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 20)))
                self.screen.blit(sub, sub.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 20)))
                pygame.display.flip()
                self.clock.tick(60)
                continue

            frame = self.get_frame()
            sx, sy = int(self.shake_x), int(self.shake_y)
            if frame:
                self.screen.blit(frame, (sx, sy))
            else:
                err = self.font_med.render("NO VIDEO SIGNAL", True, RED)
                self.screen.blit(err, err.get_rect(center=(WINDOW_W//2, WINDOW_H//2)))

            # Per-map atmosphere overlay (fog, scope mask, glitch, rain, etc.)
            self.draw_map_overlay(dt)

            # NPC detection overlays
            self.draw_npc_overlays()

            # Shoot flash (screen-wide muzzle brightness)
            self.draw_shoot_flash()

            # Vignette + scanlines + film grain
            self.screen.blit(self._vignette, (0, 0))
            self.screen.blit(self._scanlines, (0, 0))
            self._draw_grain(dt)

            # Muzzle flash (at crosshair, no gun model)
            self.draw_muzzle_flash()

            # HUD
            self.draw_corner_brackets()
            self.draw_exit_button()
            self.draw_hud_title()
            self.draw_crosshair()
            self.draw_hit_markers()
            self.draw_health_bar()
            self.draw_ammo()
            self.draw_score()
            self.draw_kill_feed()
            self.draw_fps()
            self.draw_reload_hint()
            self.draw_serial_status()
            self.draw_volume_slider()

            # Re-zero flash
            if self._rezero_flash > 0:
                alpha = min(1.0, self._rezero_flash)
                col   = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
                txt   = self.font_med.render("AIM ZEROED", True, col)
                self.screen.blit(txt, txt.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 60)))

            # Damage flash (on top of everything)
            self.draw_damage_flash()

            # END state — show debrief overlay on top of frozen frame
            if self.state == "END":
                self._draw_end_screen()

            pygame.display.flip()
            self.clock.tick(60)

        self._capture.release()
        pygame.quit()
        print("[*] Game closed.")


if __name__ == "__main__":
    game = FPSGame()
    game.run()
