import cv2
import mediapipe as mp
import random
import time
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import ImageFont, ImageDraw, Image

# ---------------- MODEL ---------------- #
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

cv2.namedWindow("RPS Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("RPS Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---------------- CONFIG ---------------- #
ROUND_COUNTDOWN = 3
RESULT_TIME = 2

# Standard button dimensions & positions (within camera panel)
BTNS_Y1, BTNS_Y2 = 10, 45
BTN_START_X = (5, 85)
BTN_RESET_X = (92, 172)
BTN_EXIT_X = (180, 255)

# Colors (BGR format)
COLOR_START = (0, 200, 0)    # Green
COLOR_RESET = (0, 165, 255)  # Orange/Gold
COLOR_EXIT = (0, 0, 200)     # Red
COLOR_HOVER = (255, 255, 255) # White glow for hover
# ---------------- STATE ---------------- #
player_score = 0
computer_score = 0

player_move = ""
computer_move = ""
result_text = ""

playing = False
show_result = False

start_time = 0
result_time = 0

anim_phase = 0
last_count = 3
countdown = 0

# ---------------- IMAGES ---------------- #
rock_img = cv2.imread("rock.png", cv2.IMREAD_UNCHANGED)
paper_img = cv2.imread("paper.png", cv2.IMREAD_UNCHANGED)
scissors_img = cv2.imread("scissors.png", cv2.IMREAD_UNCHANGED)

if rock_img is None or paper_img is None or scissors_img is None:
    print("❌ Missing images")
    exit()

# ---------------- FUNCTIONS ---------------- #
def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def decide_winner(player, computer):

    if player == "Unknown":
        return "Invalid Gesture"

    if player == computer:
        return "Draw"

    if player == "Rock":
        return "You Win" if computer == "Scissors" else "Computer Wins"

    if player == "Paper":
        return "You Win" if computer == "Rock" else "Computer Wins"

    if player == "Scissors":
        return "You Win" if computer == "Paper" else "Computer Wins"

def detect_rps(landmarks):
    """Detects Rock, Paper, or Scissors based on finger positions."""
    fingers = []
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky

    for tip in tips:
        # If tip is above the lower joint, it's 'up'
        fingers.append(1 if landmarks[tip].y < landmarks[tip-2].y else 0)

    if fingers == [0,0,0,0]: return "Rock"
    if fingers == [1,1,1,1]: return "Paper"
    if fingers == [1,1,0,0]: return "Scissors"
    return "Unknown"
    
    
def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]

    if overlay.shape[2] < 4:
        bg[y:y+h, x:x+w] = overlay
        return bg

    alpha = overlay[:,:,3]/255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * overlay[:,:,c] +
            (1-alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

def draw_panel(frame, x1, y1, x2, y2, color, alpha=0.12, radius=24):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

def draw_panel_border(frame, x1, y1, x2, y2, color, thickness=2, radius=24):
    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)
    return frame


def get_img(move):
    return {
        "Rock": rock_img,
        "Paper": paper_img,
        "Scissors": scissors_img
    }.get(move)

# ---------------- MAIN LOOP ---------------- #
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------- HAND DETECTION ---------- #
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture = "Unknown"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            gesture = detect_rps(hand_landmarks)

    # ---------- GAME LOGIC ---------- #
    if playing:
        elapsed = time.time() - start_time
        countdown = max(0, ROUND_COUNTDOWN - int(elapsed))  

        if countdown != last_count:
            anim_phase = 0
            last_count = countdown

        if countdown == 0:
            if not show_result:
                if gesture != "Unknown":
                    player_move = gesture
                    computer_move = get_computer_choice()
                    result_text = decide_winner(player_move, computer_move)

                    if result_text == "You Win":
                        player_score += 1
                    elif result_text == "Computer Wins":
                        computer_score += 1
                else:
                    result_text = "Invalid Move"

                show_result = True
                result_time = time.time()

            playing = False

    if show_result and time.time() - result_time > RESULT_TIME:
        show_result = False
        player_move = ""
        computer_move = ""
        result_text = ""

    # ================= PIXEL PERFECT UI ================= #

    ui = 255 * np.ones((h, w, 3), dtype=np.uint8)

    # ---------- SAFE MARGINS ---------- #
    margin = 60
    panel_w = 260
    panel_h = 260
    top_offset = 120

    # ---------- TITLE ---------- #
    cv2.putText(ui, "ROCK  -  PAPER  -  SCISSOR",
                (w//2 - 240, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (180, 90, 40), 2, cv2.LINE_AA)

    # ---------- SCORE ---------- #
    score = f"{computer_score} : {player_score}"
    cv2.putText(ui, score,
                (w//2 - 40, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (200, 0, 200), 2, cv2.LINE_AA)

    # ---------- PANEL POSITIONS ---------- #

    # AI (left)
    ai_x1 = margin
    ai_y1 = top_offset
    ai_x2 = ai_x1 + panel_w
    ai_y2 = ai_y1 + panel_h

    # PLAYER (right)
    pl_x2 = w - margin
    pl_y1 = top_offset
    pl_x1 = pl_x2 - panel_w
    pl_y2 = pl_y1 + panel_h

    # ---------- PANEL HEADERS ---------- #

    # AI header bar
    cv2.rectangle(ui, (ai_x1, ai_y1 - 30), (ai_x2, ai_y1),
                (180, 50, 200), -1)

    cv2.putText(ui, "AI",
                (ai_x1 + 100, ai_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2, cv2.LINE_AA)

    # PLAYER header bar
    cv2.rectangle(ui, (pl_x1, pl_y1 - 30), (pl_x2, pl_y1),
                (50, 200, 100), -1)

    cv2.putText(ui, "PLAYER",
                (pl_x1 + 70, pl_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2, cv2.LINE_AA)

    # ---------- PANEL BORDERS ---------- #

    cv2.rectangle(ui, (ai_x1, ai_y1), (ai_x2, ai_y2),
                (180, 50, 200), 3)

    cv2.rectangle(ui, (pl_x1, pl_y1), (pl_x2, pl_y2),
                (50, 200, 100), 3)

    # ---------- CENTER DIVIDER ---------- #

    cv2.line(ui,
            (w//2, ai_y1),
            (w//2, ai_y2),
            (180,180,180), 1)


    # ---------- CAMERA INSIDE PLAYER PANEL ---------- #

    cam_w = pl_x2 - pl_x1
    cam_h = pl_y2 - pl_y1

    resized = cv2.resize(frame, (cam_w, cam_h))
    # ---------- BUTTON RENDERING & INTERACTION ---------- #
    # Draw Buttons & Check Hovers
    active_hover = None
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Map index finger tip (8) to camera panel dimensions
            idx_x = int(hand_landmarks[8].x * cam_w)
            idx_y = int(hand_landmarks[8].y * cam_h)

            # Check collision
            if BTNS_Y1 < idx_y < BTNS_Y2:
                if BTN_START_X[0] < idx_x < BTN_START_X[1]: active_hover = "START"
                elif BTN_RESET_X[0] < idx_x < BTN_RESET_X[1]: active_hover = "RESET"
                elif BTN_EXIT_X[0] < idx_x < BTN_EXIT_X[1]: active_hover = "EXIT"

    # Start Button logic
    color = COLOR_HOVER if active_hover == "START" else COLOR_START
    cv2.rectangle(resized, (BTN_START_X[0], BTNS_Y1), (BTN_START_X[1], BTNS_Y2), color, -1)
    cv2.putText(resized, "START", (BTN_START_X[0]+12, BTNS_Y1+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if active_hover == "START" else (255,255,255), 2)
    if active_hover == "START" and not playing:
        playing, start_time, show_result = True, time.time(), False
        player_move, computer_move, result_text = "", "", ""

    # Reset Button logic
    color = COLOR_HOVER if active_hover == "RESET" else COLOR_RESET
    cv2.rectangle(resized, (BTN_RESET_X[0], BTNS_Y1), (BTN_RESET_X[1], BTNS_Y2), color, -1)
    cv2.putText(resized, "RESET", (BTN_RESET_X[0]+12, BTNS_Y1+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if active_hover == "RESET" else (255,255,255), 2)
    if active_hover == "RESET":
        player_score, computer_score, playing, show_result = 0, 0, False, False

    # Exit Button logic
    color = COLOR_HOVER if active_hover == "EXIT" else COLOR_EXIT
    cv2.rectangle(resized, (BTN_EXIT_X[0], BTNS_Y1), (BTN_EXIT_X[1], BTNS_Y2), color, -1)
    cv2.putText(resized, "EXIT", (BTN_EXIT_X[0]+18, BTNS_Y1+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) if active_hover == "EXIT" else (255,255,255), 2)
    if active_hover == "EXIT":
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # ---------------- HAND DETECTION UI (PRO) ---------------- #

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            xs, ys = [], []

            # 🔹 draw joints (clean + small)
            for i, lm in enumerate(hand_landmarks):
                x = int(lm.x * cam_w)
                y = int(lm.y * cam_h)

                xs.append(x)
                ys.append(y)

                cv2.circle(resized, (x, y), 3, (0, 255, 255), -1)

            # 🔹 draw connections (like skeleton)
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]

            for c in connections:
                x1 = int(hand_landmarks[c[0]].x * cam_w)
                y1 = int(hand_landmarks[c[0]].y * cam_h)
                x2 = int(hand_landmarks[c[1]].x * cam_w)
                y2 = int(hand_landmarks[c[1]].y * cam_h)

                cv2.line(resized, (x1,y1), (x2,y2),
                        (255, 255, 255), 1, cv2.LINE_AA)

            # 🔥 LASER BOX
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            # glow
            cv2.rectangle(resized, (x1-4, y1-4), (x2+4, y2+4),
                        (255, 0, 255), 2)

            # sharp
            cv2.rectangle(resized, (x1, y1), (x2, y2),
                        (255, 0, 255), 1, cv2.LINE_AA)

            # 📍 coordinates
            cv2.putText(resized,
                        f"X:{x1} Y:{y1}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1, cv2.LINE_AA)

            # 🔹 highlight fingertips
            tips = [8, 12, 16, 20]
            for tip in tips:
                tx = int(hand_landmarks[tip].x * cam_w)
                ty = int(hand_landmarks[tip].y * cam_h)

                cv2.circle(resized, (tx, ty), 6, (0, 255, 0), -1)
    
    # now place into UI
    ui[pl_y1:pl_y2, pl_x1:pl_x2] = resized
    
    
    # ---------- AI IMAGE ---------- #

    if computer_move:
        comp_img = get_img(computer_move)

        if comp_img is not None:
            comp_img = cv2.resize(comp_img, (180,180))

            cx = ai_x1 + (panel_w - 180)//2
            cy = ai_y1 + (panel_h - 180)//2

            ui[cy:cy+180, cx:cx+180] = comp_img[:, :, :3]

    # ---------- COUNTDOWN ---------- #

    if playing:
        if countdown > 0:
            cv2.putText(ui, str(countdown),
                        (w//2 - 20, ai_y2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 3, cv2.LINE_AA)

        elif countdown == 0:
            cv2.putText(ui, "GO!",
                        (w//2 - 40, ai_y2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,180,0), 3, cv2.LINE_AA)

    # ---------- RESULT ---------- #

    # (Logic moved to show_result block below for neatness)

    # (Footer key guide removed as requested)
    
    # (Redundant start logic removed for cleaner code)

    # ---------- FINAL FRAME ---------- #
    frame = ui
    
    # ---------------- COMPACT RESULT BOX ---------------- #

    if show_result:

        font = cv2.FONT_HERSHEY_COMPLEX
        scale = 0.8
        thickness = 2

        # get text size
        text_size = cv2.getTextSize(result_text, font, scale, thickness)[0]

        # padding
        pad_x = 20
        pad_y = 15

        box_w = text_size[0] + pad_x * 2
        box_h = text_size[1] + pad_y * 2

        # center position
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        # background
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)

        # text
        text_x = x1 + pad_x
        text_y = y1 + box_h - pad_y

        cv2.putText(frame, result_text,
                (text_x, text_y),
                font,
                scale,
                (0, 0, 0),
                thickness)
        
    # ---------- SHOW ---------- #
    cv2.imshow("RPS Game", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------- VIRTUAL BUTTON LOGIC ---------- #
    button_hovered = False
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()