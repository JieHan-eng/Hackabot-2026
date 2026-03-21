# visualiser.py — 3D Gun Controller Visualiser
# Run on your PC (NOT on the Pico)
# Reads serial data from Pico and renders a live 3D view
#
# Install dependencies:
#   pip install pyserial pygame PyOpenGL
#
# Usage:
#   python visualiser.py
#   (or specify port: python visualiser.py COM3)
#   (or on Mac/Linux: python visualiser.py /dev/tty.usbmodem1234)

import sys
import json
import time
import serial
import serial.tools.list_ports
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# ============================================================
# SERIAL CONNECTION
# ============================================================

def find_pico_port():
    """Auto-detect the Pico's serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or '').lower()
        if 'pico' in desc or 'board' in desc or 'usb serial' in desc or 'usbmodem' in desc:
            return p.device
    # Fallback: show available ports
    if ports:
        print("Available ports:")
        for p in ports:
            print(f"  {p.device} - {p.description}")
        return ports[0].device
    return None


def connect_serial(port=None, baud=115200):
    """Connect to the Pico over serial."""
    if port is None:
        port = find_pico_port()
    if port is None:
        print("ERROR: No serial port found. Is the Pico plugged in?", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"Connecting to {port} at {baud} baud...", file=sys.stderr, flush=True)
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(0.5)  # let it settle
        ser.reset_input_buffer()
        print(f"Connected to {port}", file=sys.stderr, flush=True)
        return ser
    except Exception as e:
        print(f"ERROR: Could not open {port}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


# ============================================================
# 3D RENDERING
# ============================================================

def draw_gun(yaw, roll):
    """Draw a simplified gun shape that rotates with yaw and roll."""
    glPushMatrix()
    
    # Apply rotations: yaw around Y axis, roll around Z axis
    glRotatef(yaw, 0, 1, 0)   # left/right rotation
    glRotatef(roll, 0, 0, 1)  # up/down tilt
    
    # Gun body (barrel pointing forward along X)
    glColor3f(0.45, 0.45, 0.50)
    
    # Main body block
    draw_box(-0.3, -0.15, -0.12, 0.8, 0.3, 0.24)
    
    # Barrel
    glColor3f(0.35, 0.35, 0.40)
    draw_box(0.5, -0.06, -0.06, 1.2, 0.12, 0.12)
    
    # Barrel tip (slightly wider)
    glColor3f(0.30, 0.30, 0.35)
    draw_box(1.15, -0.08, -0.08, 0.15, 0.16, 0.16)
    
    # Handle (grip going down)
    glColor3f(0.3, 0.25, 0.2)
    draw_box(-0.05, -0.55, -0.08, 0.2, 0.45, 0.16)
    
    # Sight (top bump)
    glColor3f(0.5, 0.5, 0.55)
    draw_box(0.3, 0.15, -0.03, 0.15, 0.08, 0.06)
    
    # Forward direction indicator (small red tip)
    glColor3f(1.0, 0.3, 0.2)
    draw_box(1.3, -0.03, -0.03, 0.05, 0.06, 0.06)
    
    glPopMatrix()


def draw_box(x, y, z, w, h, d):
    """Draw a simple box at position (x,y,z) with size (w,h,d)."""
    x2, y2, z2 = x + w, y + h, z + d
    
    glBegin(GL_QUADS)
    # Front
    glVertex3f(x, y, z2); glVertex3f(x2, y, z2)
    glVertex3f(x2, y2, z2); glVertex3f(x, y2, z2)
    # Back
    glVertex3f(x, y, z); glVertex3f(x, y2, z)
    glVertex3f(x2, y2, z); glVertex3f(x2, y, z)
    # Top
    glVertex3f(x, y2, z); glVertex3f(x, y2, z2)
    glVertex3f(x2, y2, z2); glVertex3f(x2, y2, z)
    # Bottom
    glVertex3f(x, y, z); glVertex3f(x2, y, z)
    glVertex3f(x2, y, z2); glVertex3f(x, y, z2)
    # Right
    glVertex3f(x2, y, z); glVertex3f(x2, y2, z)
    glVertex3f(x2, y2, z2); glVertex3f(x2, y, z2)
    # Left
    glVertex3f(x, y, z); glVertex3f(x, y, z2)
    glVertex3f(x, y2, z2); glVertex3f(x, y2, z)
    glEnd()
    
    # Wireframe edges
    glColor3f(0.15, 0.15, 0.15)
    glBegin(GL_LINE_LOOP)
    glVertex3f(x, y, z2); glVertex3f(x2, y, z2)
    glVertex3f(x2, y2, z2); glVertex3f(x, y2, z2)
    glEnd()
    glBegin(GL_LINE_LOOP)
    glVertex3f(x, y, z); glVertex3f(x2, y, z)
    glVertex3f(x2, y2, z); glVertex3f(x, y2, z)
    glEnd()


def draw_grid():
    """Draw a reference grid on the ground plane."""
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in range(-5, 6):
        glVertex3f(i, -0.8, -5); glVertex3f(i, -0.8, 5)
        glVertex3f(-5, -0.8, i); glVertex3f(5, -0.8, i)
    glEnd()
    
    # Forward direction line (red)
    glColor3f(0.8, 0.2, 0.2)
    glBegin(GL_LINES)
    glVertex3f(0, -0.8, 0); glVertex3f(3, -0.8, 0)
    glEnd()


def draw_crosshair(screen_w, screen_h, yaw, roll):
    """Draw a 2D crosshair overlay showing where the gun is pointing."""
    # Switch to 2D overlay mode
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, screen_w, screen_h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    
    # Crosshair position (center + offset based on yaw/roll)
    cx = screen_w // 2 + yaw * 3
    cy = screen_h // 2 - roll * 3
    size = 15
    
    glColor3f(0.2, 1.0, 0.3)
    glLineWidth(2)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size)
    glEnd()
    
    # Outer circle (approximate with lines)
    glBegin(GL_LINE_LOOP)
    for a in range(0, 360, 10):
        rad = a * math.pi / 180
        glVertex2f(cx + math.cos(rad) * size * 1.5, cy + math.sin(rad) * size * 1.5)
    glEnd()
    glLineWidth(1)
    
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def draw_hud(screen, data):
    """Draw text HUD overlay with pygame."""
    font = pygame.font.SysFont('consolas', 18)
    
    lines = [
        f"YAW:  {data['yaw']:>7.1f} deg  (rotate L/R)",
        f"ROLL: {data['roll']:>7.1f} deg  (tilt U/D)",
        f"",
        f"STEER:    {data['steer']:>4d}  (joystick 1 X)",
        f"THROTTLE: {data['throttle']:>4d}  (joystick 1 Y)",
        f"",
        f"FIRE: {'>>> FIRING <<<' if data['fire'] else '---'}",
    ]
    
    y = 10
    for line in lines:
        if 'FIRING' in line:
            color = (255, 80, 60)
        else:
            color = (200, 200, 200)
        surface = font.render(line, True, color)
        screen.blit(surface, (10, y))
        y += 22
    
    # Controls help at bottom
    help_lines = [
        "[R] Reset yaw to 0    [Q/ESC] Quit",
    ]
    y = screen.get_height() - 30
    for line in help_lines:
        surface = font.render(line, True, (120, 120, 120))
        screen.blit(surface, (10, y))
        y += 22


# ============================================================
# MAIN
# ============================================================

def main():
    # Parse command line for port
    port = sys.argv[1] if len(sys.argv) > 1 else None
    ser = connect_serial(port)
    
    # Init pygame + OpenGL
    pygame.init()
    screen_w, screen_h = 900, 600
    screen = pygame.display.set_mode((screen_w, screen_h), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Gun Controller 3D Visualiser")
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.12, 0.12, 0.14, 1)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, screen_w / screen_h, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Camera position (looking at the gun from a 3/4 angle)
    cam_x, cam_y, cam_z = 2.5, 1.5, 2.5
    
    # Current data
    data = {'yaw': 0, 'roll': 0, 'steer': 0, 'throttle': 0, 'fire': 0}
    
    clock = pygame.time.Clock()
    running = True
    
    print("\nVisualiser running! Move the gun controller to see it respond.")
    print("Press R to reset yaw, Q or ESC to quit.\n")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key in (K_q, K_ESCAPE):
                    running = False
                elif event.key == K_r:
                    # Send reset command (the Pico will see this if we add it later)
                    print("Yaw reset requested")
        
        # Read serial data (non-blocking, read all available lines)
        for _ in range(50):  # read up to 50 lines per frame to drain buffer
            if not ser.in_waiting:
                break
            try:
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    parsed = json.loads(line)
                    data.update(parsed)
            except (json.JSONDecodeError, ValueError):
                pass  # skip malformed lines
        
        # Render 3D scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0)
        
        draw_grid()
        draw_gun(data['yaw'], data['roll'])
        draw_crosshair(screen_w, screen_h, data['yaw'], data['roll'])
        
        # Fire flash effect
        if data['fire']:
            glClearColor(0.25, 0.1, 0.08, 1)
        else:
            glClearColor(0.12, 0.12, 0.14, 1)
        
        # Draw HUD: read GL framebuffer, blit with text overlay, single flip
        gl_pixels = glReadPixels(0, 0, screen_w, screen_h, GL_RGB, GL_UNSIGNED_BYTE)
        pg_surface = pygame.image.fromstring(gl_pixels, (screen_w, screen_h), 'RGB', True)
        screen.blit(pg_surface, (0, 0))

        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        draw_hud(overlay, data)
        screen.blit(overlay, (0, 0))

        pygame.display.flip()
        
        clock.tick(60)
    
    ser.close()
    pygame.quit()
    print("Visualiser closed.")


if __name__ == '__main__':
    main()
