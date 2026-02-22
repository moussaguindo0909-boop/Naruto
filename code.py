import cv2
import mediapipe as mp
import numpy as np
import sys
import math
import random
import time
from collections import deque

sys.stdout.reconfigure(encoding='utf-8')

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)

# NOUVEAU : Configuration du Face Mesh pour détecter les yeux
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  refine_landmarks=True, # Nécessaire pour avoir les iris précis
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5)

# ── Constants ─────────────────────────────────────────────────────────────────
TIPS = [4, 8, 12, 16, 20]
BASES = [2, 6, 10, 14, 18]
MAX_PARTICLES = 500
SMOOTHING_HISTORY = 8

# Indices des landmarks pour les yeux (Iris)
LEFT_EYE_CENTER = 468   # Centre de l'iris gauche
RIGHT_EYE_CENTER = 473  # Centre de l'iris droit

# ── Finger Detection ──────────────────────────────────────────────────────────
def get_fingers(lm, label):
    fingers = []
    # Thumb
    if label == "Right":
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)
    # Others
    for tip, base in zip(TIPS[1:], BASES[1:]):
        fingers.append(1 if lm[tip].y < lm[base].y else 0)
    return fingers

# ── Gesture Recognition ───────────────────────────────────────────────────────
def recognize_gesture(fingers_list, landmarks_list):
    if not fingers_list:
        return None

    if len(fingers_list) == 2:
        f1, f2 = fingers_list[0], fingers_list[1]
        w1 = landmarks_list[0].landmark[0]
        w2 = landmarks_list[1].landmark[0]
        dist = math.sqrt((w1.x - w2.x)**2 + (w1.y - w2.y)**2)

        if f1 == [0,0,0,0,0] and f2 == [0,0,0,0,0] and dist < 0.2:
            return "AMATERASU"
        if (f1[1] == 1 and f1[2] == 1 and f1[3] == 0 and f1[4] == 0 and
            f2[1] == 1 and f2[2] == 1 and f2[3] == 0 and f2[4] == 0):
            return "SUSANOO"

    f = fingers_list[0]
    if f == [0, 0, 0, 0, 0]: return "KATON"
    if f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0: return "CHIDORI"
    if f == [1, 1, 1, 1, 1]: return "RASENGAN"
    if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 1: return "SHARINGAN"
    if f[1] == 1 and f[2] == 0 and f[3] == 0 and f[4] == 0: return "TAJUU_KAGE"
    if f == [1, 0, 0, 0, 0]: return "AMATERASU" 
    if f[0] == 0 and f[1] == 1 and f[2] == 1 and f[3] == 1 and f[4] == 1: return "BYAKUGAN"
    if f[0] == 0 and f[1] == 0 and f[2] == 1 and f[3] == 1 and f[4] == 1: return "SUITON"
    return None

# ── Particle System ───────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, vx=None, vy=None, size=None, life=None, ptype="normal"):
        self.x = float(x)
        self.y = float(y)
        self.color = list(color)
        self.vx = vx if vx is not None else random.uniform(-4, 4)
        self.vy = vy if vy is not None else random.uniform(-8, -1)
        self.size = size if size is not None else random.randint(3, 10)
        self.life = life if life is not None else random.uniform(0.5, 1.5)
        self.born = time.time()
        self.ptype = ptype 
        self.trail = deque(maxlen=5)

    @property
    def alive(self):
        return (time.time() - self.born) < self.life

    def update(self):
        self.trail.append((self.x, self.y))
        if self.ptype == 'fire':
            self.vx += random.uniform(-0.5, 0.5)
            self.vy -= 0.1 
            self.color[0] = max(0, self.color[0] - 5)
        elif self.ptype == 'spark':
            self.vy += 0.5
            self.size *= 0.95
        elif self.ptype == 'smoke':
            self.vy -= 0.5
            self.vx += random.uniform(-0.5, 0.5)
            self.size *= 1.05
            self.color = [200, 200, 200] 
        self.x += self.vx
        self.y += self.vy

particles = []

def manage_particles(frame, canvas_glow):
    global particles
    particles = [p for p in particles if p.alive]
    if len(particles) > MAX_PARTICLES:
        particles = particles[-MAX_PARTICLES:]
    for p in particles:
        age = (time.time() - p.born) / p.life
        alpha = 1.0 - age
        if len(p.trail) > 1:
            for i in range(len(p.trail) - 1):
                pt1 = (int(p.trail[i][0]), int(p.trail[i][1]))
                pt2 = (int(p.trail[i+1][0]), int(p.trail[i+1][1]))
                cv2.line(frame, pt1, pt2, p.color, 1)
        px, py, ps = int(p.x), int(p.y), max(1, int(p.size * alpha))
        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(canvas_glow, (px, py), ps, p.color, -1)
            cv2.circle(frame, (px, py), max(1, ps//2), (255,255,255), -1)
        p.update()

# ── Animation Functions ────────────────────────────────────────────────────────

def draw_fire_advanced(frame, cx, cy, canvas_glow):
    cv2.circle(canvas_glow, (cx, cy), 50, (0, 100, 255), -1)
    for _ in range(10):
        particles.append(Particle(cx + random.randint(-20, 20), cy + random.randint(-10, 10), color=(random.randint(0, 50), random.randint(50, 200), 255), vx=random.uniform(-3, 3), vy=random.uniform(-10, -3), size=random.randint(10, 25), life=random.uniform(0.4, 0.8), ptype='fire'))
    if random.random() < 0.2:
        particles.append(Particle(cx, cy, (50,50,50), vy=-3, size=20, life=1.5, ptype='smoke'))

def draw_chidori_advanced(frame, cx, cy, canvas_glow, t):
    cv2.circle(canvas_glow, (cx, cy), 60, (255, 180, 0), -1)
    cv2.circle(frame, (cx, cy), 15, (255, 255, 255), -1)
    for _ in range(12):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(60, 150)
        x2 = int(cx + math.cos(angle) * length)
        y2 = int(cy + math.sin(angle) * length)
        pts = [(cx, cy)]
        for i in range(1, 6):
            px = cx + int((x2 - cx) * i / 6) + random.randint(-20, 20)
            py = cy + int((y2 - cy) * i / 6) + random.randint(-20, 20)
            pts.append((px, py))
        pts.append((x2, y2))
        for i in range(len(pts) - 1):
            cv2.line(canvas_glow, pts[i], pts[i+1], (255, 255, 255), 3)
            cv2.line(frame, pts[i], pts[i+1], (200, 200, 255), 1)
    if random.random() < 0.5:
        angle = random.uniform(0, 2 * math.pi)
        vel = random.uniform(5, 15)
        particles.append(Particle(cx, cy, (255, 255, 255), vx=math.cos(angle)*vel, vy=math.sin(angle)*vel, size=3, life=0.3, ptype='spark'))

def draw_rasengan_advanced(frame, cx, cy, canvas_glow, t):
    for i in range(5):
        radius = int(80 - i * 15 + 5 * math.sin(t * 5 + i))
        cv2.circle(canvas_glow, (cx, cy), radius, (255, 150, 0), -1)
    for i in range(100):
        angle = (t * 10 + i * 5) * math.pi / 180
        r = i * 0.8
        rot_angle = angle + t * 5 
        px = int(cx + r * math.cos(rot_angle))
        py = int(cy + r * math.sin(rot_angle))
        color = (255, int(150 + i), int(50 + i*2))
        cv2.circle(frame, (px, py), 2, color, -1)
        cv2.circle(canvas_glow, (px, py), 4, color, -1)
    cv2.circle(frame, (cx, cy), 15, (255, 255, 255), -1)
    draw_jutsu_text(frame, "RASENGAN !", (200, 100, 0))

# --- ANIMATION SHARINGAN (MODIFIÉE POUR LES YEUX) ---
def draw_sharingan_advanced(frame, eye_positions, canvas_glow, t):
    """
    eye_positions: liste de tuples (x, y) pour chaque œil détecté
    """
    for (ex, ey) in eye_positions:
        # Rayon adaptatif pour l'œil
        radius = 30 
        
        # Lueur rouge autour de l'œil
        cv2.circle(canvas_glow, (ex, ey), int(radius * 1.5), (0, 0, 200), -1)
        
        # Fond rouge de l'œil (Sharingan)
        cv2.circle(frame, (ex, ey), radius, (0, 0, 150), -1)
        
        # Anneaux noirs
        cv2.circle(frame, (ex, ey), int(radius * 0.7), (0, 0, 0), 2)
        
        # Pupille noire
        cv2.circle(frame, (ex, ey), int(radius * 0.4), (0, 0, 0), -1)
        
        # Tomoes tournants (3 par œil)
        for i in range(3):
            angle = t * 4 + i * (2 * math.pi / 3)
            dist = radius * 0.55 # Distance du centre
            tx = int(ex + dist * math.cos(angle))
            ty = int(ey + dist * math.sin(angle))
            
            # Dessiner tomoe
            cv2.circle(frame, (tx, ty), int(radius * 0.15), (0, 0, 0), -1)
            # Queue du tomoe
            tail_x = int(tx - (radius * 0.2) * math.cos(angle - 0.5))
            tail_y = int(ty - (radius * 0.2) * math.sin(angle - 0.5))
            cv2.line(frame, (tx, ty), (tail_x, tail_y), (0, 0, 0), int(radius * 0.1))

    draw_jutsu_text(frame, "SHARINGAN !", (0, 0, 255))

# --- ANIMATION BYAKUGAN (MODIFIÉE POUR LES YEUX) ---
def draw_byakugan_advanced(frame, eye_positions, canvas_glow, t):
    for (ex, ey) in eye_positions:
        radius = 30
        
        # 1. Veines qui rayonnent de l'œil
        for i in range(0, 360, 15):
            angle = (i + t * 10) * math.pi / 180
            length = radius * 3 + random.randint(-5, 5)
            x2 = int(ex + length * math.cos(angle))
            y2 = int(ey + length * math.sin(angle))
            # Veines légèrement visibles
            cv2.line(frame, (ex, ey), (x2, y2), (180, 160, 160), 1)
        
        # 2. Œil blanc laiteux
        cv2.circle(frame, (ex, ey), radius, (240, 240, 240), -1)
        
        # 3. Pupille très grande et sombre (pas d'iris visible)
        cv2.circle(frame, (ex, ey), int(radius * 0.6), (40, 40, 40), -1)
        
        # Lueur blanche intense
        cv2.circle(canvas_glow, (ex, ey), int(radius * 1.2), (200, 200, 200), -1)

    draw_jutsu_text(frame, "BYAKUGAN !", (255, 255, 255))

def draw_amaterasu_advanced(frame, cx, cy, canvas_glow):
    for _ in range(8):
        particles.append(Particle(cx + random.randint(-40, 40), cy + random.randint(-20, 20), color=(10, 10, 10), vx=random.uniform(-1, 1), vy=random.uniform(-4, -1), size=random.randint(20, 50), life=random.uniform(1.0, 2.0), ptype='smoke'))
    cv2.circle(canvas_glow, (cx, cy), 50, (150, 0, 150), -1)
    cv2.circle(frame, (cx, cy), 30, (0, 0, 0), -1)
    draw_jutsu_text(frame, "AMATERASU !", (100, 0, 100))

def draw_suiton_advanced(frame, cx, cy, canvas_glow, t):
    h, w = frame.shape[:2]
    for _ in range(15):
        particles.append(Particle(cx + random.randint(-100, 100), cy + 50, color=(255, 150, 50), vx=random.uniform(-4, 4), vy=random.uniform(-12, -6), size=random.randint(5, 12), life=1.0, ptype='spark'))
    pts = []
    for x in range(0, w, 30):
        y = int(h/2 + 50 * math.sin(x * 0.02 + t * 4))
        pts.append([x, y])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], False, (255, 200, 100), 5)
    cv2.fillPoly(canvas_glow, [pts], (255, 200, 100))
    draw_jutsu_text(frame, "SUITON !", (255, 150, 0))

def draw_susanoo_advanced(frame, cx, cy, canvas_glow, t):
    h, w = frame.shape[:2]
    purple = (180, 0, 180)
    white = (255, 255, 255)
    cv2.line(canvas_glow, (cx, 0), (cx, h), purple, 10)
    rib_cycle = math.sin(t * 2) * 20
    for i in range(4):
        y_offset = cy - 150 + i * 80
        rib_width = 200 - i * 20
        cv2.line(frame, (cx, y_offset), (cx - rib_width, y_offset + 50 + int(rib_cycle)), purple, 8)
        cv2.line(canvas_glow, (cx, y_offset), (cx - rib_width, y_offset + 50 + int(rib_cycle)), white, 15)
        cv2.line(frame, (cx, y_offset), (cx + rib_width, y_offset + 50 + int(rib_cycle)), purple, 8)
        cv2.line(canvas_glow, (cx, y_offset), (cx + rib_width, y_offset + 50 + int(rib_cycle)), white, 15)
    if random.random() < 0.3:
        particles.append(Particle(random.randint(0, w), random.randint(0, h), color=(150, 50, 150), vy=-2, size=20, life=1.0, ptype='smoke'))
    draw_jutsu_text(frame, "SUSANOO !", (200, 0, 200))

def draw_tajuu_advanced(frame, t):
    h, w = frame.shape[:2]
    M_left = np.float32([[1, 0, -180], [0, 1, 0]])
    clone_left = cv2.warpAffine(frame, M_left, (w, h))
    M_right = np.float32([[1, 0, 180], [0, 1, 0]])
    clone_right = cv2.warpAffine(frame, M_right, (w, h))
    cv2.addWeighted(frame, 0.7, clone_left, 0.3, 0, frame)
    cv2.addWeighted(frame, 0.7, clone_right, 0.3, 0, frame)
    global particles
    if random.random() < 0.4:
        pos_x = random.choice([w//2 - 180, w//2 + 180])
        pos_y = random.randint(h//3, h//2 + 50)
        particles.append(Particle(pos_x, pos_y, color=(240, 240, 240), vx=random.uniform(-1, 1), vy=random.uniform(-2, 1), size=random.randint(30, 60), life=random.uniform(0.6, 1.2), ptype='smoke'))
    draw_jutsu_text(frame, "TAJUU KAGE BUNSHIN !", (255, 255, 255))

def draw_jutsu_text(frame, text, color):
    h, w = frame.shape[:2]
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, scale, thickness)
    x = (w - tw) // 2
    y = h - 50
    cv2.putText(frame, text, (x+3, y+3), cv2.FONT_HERSHEY_TRIPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(frame, text, (x, y),     cv2.FONT_HERSHEY_TRIPLEX, scale, color,   thickness)

# ── Main Loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("System Ready -- Jutsu Mode Activated  |  Appuyez sur ESC pour quitter")
gesture_history = deque(maxlen=SMOOTHING_HISTORY)
t = 0.0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    
    glow_layer = np.zeros_like(frame)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Traitement des Mains
    result = hands.process(rgb)
    fingers_list = []
    landmarks_list = []
    centers = []

    if result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_info.classification[0].label
            lm = hand_lm.landmark
            f = get_fingers(lm, label)
            fingers_list.append(f)
            landmarks_list.append(hand_lm)
            wrist = lm[0]
            centers.append((int(wrist.x * w), int(wrist.y * h)))
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    cx, cy = w // 2, h // 2
    if centers:
        cx = int(sum([c[0] for c in centers]) / len(centers))
        cy = int(sum([c[1] for c in centers]) / len(centers))

    # 2. Traitement du Visage (pour les yeux)
    face_result = face_mesh.process(rgb)
    eye_positions = []
    
    if face_result.multi_face_landmarks:
        # On prend le premier visage détecté
        face_landmarks = face_result.multi_face_landmarks[0]
        
        # Récupérer les coordonnées des iris
        # Landmark 468: Œil gauche, 473: Œil droit (nécessite refine_landmarks=True)
        left_eye = face_landmarks.landmark[LEFT_EYE_CENTER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_CENTER]
        
        # Conversion en pixels
        le_x, le_y = int(left_eye.x * w), int(left_eye.y * h)
        re_x, re_y = int(right_eye.x * w), int(right_eye.y * h)
        
        eye_positions = [(le_x, le_y), (re_x, re_y)]

    # 3. Logique de Geste
    current_raw_gesture = recognize_gesture(fingers_list, landmarks_list)
    gesture_history.append(current_raw_gesture)
    
    gesture = None
    if gesture_history:
        valid_gestures = [g for g in gesture_history if g is not None]
        if valid_gestures:
            gesture = max(set(valid_gestures), key=valid_gestures.count)

    # 4. Lancement des Animations
    if gesture == "KATON":
        draw_fire_advanced(frame, cx, cy, glow_layer)
    elif gesture == "AMATERASU":
        draw_amaterasu_advanced(frame, cx, cy, glow_layer)
    elif gesture == "CHIDORI":
        draw_chidori_advanced(frame, cx, cy, glow_layer, t)
    elif gesture == "SUSANOO":
        draw_susanoo_advanced(frame, cx, cy, glow_layer, t)
    elif gesture == "RASENGAN":
        draw_rasengan_advanced(frame, cx, cy, glow_layer, t)
    elif gesture == "SHARINGAN":
        # Si on a détecté les yeux, on dessine dessus, sinon au centre
        if not eye_positions:
            eye_positions = [(cx, cy)] # Fallback
        draw_sharingan_advanced(frame, eye_positions, glow_layer, t)
    elif gesture == "TAJUU_KAGE":
        draw_tajuu_advanced(frame, t)
    elif gesture == "BYAKUGAN":
        # Si on a détecté les yeux, on dessine dessus, sinon au centre
        if not eye_positions:
            eye_positions = [(cx, cy)] # Fallback
        draw_byakugan_advanced(frame, eye_positions, glow_layer, t)
    elif gesture == "SUITON":
        draw_suiton_advanced(frame, cx, cy, glow_layer, t)

    manage_particles(frame, glow_layer)

    # Post-processing
    glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
    cv2.add(frame, glow_layer, frame)

    # UI
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "ESC: Quitter", (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Guide des gestes (coin bas-gauche)
    guide = [
        "Gestes:",
        "Poing ferme  -> KATON",
        "Index+Majeur -> CHIDORI",
        "Tous ouverts -> RASENGAN",
        "Index seul   -> TAJUU KAGE",
        "Cornes \\m/   -> SHARINGAN",
        "4 doigts     -> BYAKUGAN",
        "3 doigts     -> SUITON",
        "Pouce seul   -> AMATERASU",
        "2 mains fermees -> AMATERASU",
        "2x index+maj -> SUSANOO",
    ]
    for i, line in enumerate(guide):
        cv2.putText(frame, line, (10, h - 190 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    
    if gesture:
        cv2.rectangle(frame, (10, 10), (400, 45), (20, 20, 20), -1)
        cv2.putText(frame, f"Jutsu : {gesture}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)

    cv2.imshow("Naruto Jutsu Engine", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC pour quitter
        break

    t += 0.05

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()