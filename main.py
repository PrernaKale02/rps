import cv2
import mediapipe as mp
import random
import time
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Font
FONT = cv2.FONT_HERSHEY_DUPLEX
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

# Minimalist Dark Theme Colors (BGR)
COLOR_BG = (30, 30, 30)         # Dark Gray
COLOR_PLAYER = (250, 200, 100)  # Soft Light Blue
COLOR_AI = (100, 100, 250)      # Soft Light Red
COLOR_TEXT = (240, 240, 240)    # Off-White
COLOR_TEXT_INV = (20, 20, 20)   # Dark Gray for bright backgrounds

# Button Colors
COLOR_START = (100, 200, 100)
COLOR_RESET = (100, 150, 200)
COLOR_EXIT = (100, 100, 200)
COLOR_HOVER = (200, 200, 200)
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
score_updated = False

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

# Removed slow PIL and blur rendering functions. Using native cv2 below.
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
                else:
                    result_text = "Invalid Move"

                show_result = True
                result_time = time.time()
                score_updated = False

            playing = False

    if show_result:
        if time.time() - result_time > 1.0 and not score_updated:
            if result_text == "You Win":
                player_score += 1
            elif result_text == "Computer Wins":
                computer_score += 1
            score_updated = True

    if show_result and time.time() - result_time > RESULT_TIME + 1.0:
        show_result = False
        player_move = ""
        computer_move = ""
        result_text = ""

    # ================= MINIMAL UI ================= #
    ui = np.zeros((h, w, 3), dtype=np.uint8)
    ui[:] = COLOR_BG  # Solid dark background

    # Layout Consts
    margin, panel_w, panel_h, top_offset = 60, 260, 260, 120

    # ---------- TITLE ---------- #
    cv2.putText(ui, "ROCK  PAPER  SCISSORS", (w//2 - 180, 45), FONT, 1, COLOR_TEXT, 2, cv2.LINE_AA)
    
    # Title accent line
    cv2.line(ui, (w//2 - 190, 65), (w//2 + 190, 65), (100, 100, 100), 1, cv2.LINE_AA)

    # ---------- SCORE ---------- #
    score = f"{computer_score}   VS   {player_score}"
    cv2.putText(ui, score, (w//2 - 60, 85), FONT, 0.7, COLOR_TEXT, 1, cv2.LINE_AA)

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

    # ---------- CLEAN PANELS ---------- #
    # AI Header
    cv2.putText(ui, "AI", (ai_x1, ai_y1 - 15), FONT, 0.6, COLOR_AI, 1, cv2.LINE_AA)
    cv2.rectangle(ui, (ai_x1, ai_y1), (ai_x2, ai_y2), COLOR_AI, 1, cv2.LINE_AA)

    # PLAYER Header
    cv2.putText(ui, "PLAYER", (pl_x1, pl_y1 - 15), FONT, 0.6, COLOR_PLAYER, 1, cv2.LINE_AA)
    cv2.rectangle(ui, (pl_x1, pl_y1), (pl_x2, pl_y2), COLOR_PLAYER, 1, cv2.LINE_AA)

    # ---------- CENTER DIVIDER ---------- #
    cv2.line(ui, (w//2, ai_y1), (w//2, ai_y2), (60, 60, 60), 1)


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
    b_color = COLOR_HOVER if active_hover == "START" else COLOR_START
    cv2.rectangle(resized, (BTN_START_X[0], BTNS_Y1), (BTN_START_X[1], BTNS_Y2), b_color, -1)
    cv2.putText(resized, "START", (BTN_START_X[0]+12, BTNS_Y1+22), FONT, 0.5, COLOR_TEXT_INV, 1, cv2.LINE_AA)
    if active_hover == "START" and not playing:
        playing, start_time, show_result = True, time.time(), False
        player_move, computer_move, result_text = "", "", ""

    # Reset Button logic
    b_color = COLOR_HOVER if active_hover == "RESET" else COLOR_RESET
    cv2.rectangle(resized, (BTN_RESET_X[0], BTNS_Y1), (BTN_RESET_X[1], BTNS_Y2), b_color, -1)
    cv2.putText(resized, "RESET", (BTN_RESET_X[0]+12, BTNS_Y1+22), FONT, 0.5, COLOR_TEXT_INV, 1, cv2.LINE_AA)
    if active_hover == "RESET":
        player_score, computer_score, playing, show_result = 0, 0, False, False

    # Exit Button logic
    b_color = COLOR_HOVER if active_hover == "EXIT" else COLOR_EXIT
    cv2.rectangle(resized, (BTN_EXIT_X[0], BTNS_Y1), (BTN_EXIT_X[1], BTNS_Y2), b_color, -1)
    cv2.putText(resized, "EXIT", (BTN_EXIT_X[0]+18, BTNS_Y1+22), FONT, 0.5, COLOR_TEXT_INV, 1, cv2.LINE_AA)
    if active_hover == "EXIT":
        cap.release(); cv2.destroyAllWindows(); exit()

    # ---------------- HAND DETECTION UI (PRO) ---------------- #

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            xs, ys = [], []

            for i, lm in enumerate(hand_landmarks):
                x, y = int(lm.x * cam_w), int(lm.y * cam_h)
                xs.append(x); ys.append(y)
                cv2.circle(resized, (x, y), 2, COLOR_PLAYER, -1)

            # 🔹 draw connections cleanly
            connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
            for c in connections:
                x1, y1 = int(hand_landmarks[c[0]].x * cam_w), int(hand_landmarks[c[0]].y * cam_h)
                x2, y2 = int(hand_landmarks[c[1]].x * cam_w), int(hand_landmarks[c[1]].y * cam_h)
                cv2.line(resized, (x1,y1), (x2,y2), COLOR_PLAYER, 1, cv2.LINE_AA)
    
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
        c_text = str(countdown) if countdown > 0 else "GO!"
        c_color = (0, 0, 255) if countdown > 0 else (0, 255, 0)
        cv2.putText(ui, c_text, (w//2 - 25, ai_y2 + 50), FONT, 1.5, c_color, 2, cv2.LINE_AA)

    # ---------- RESULT ---------- #

    # (Logic moved to show_result block below for neatness)

    # (Footer key guide removed as requested)
    
    # (Redundant start logic removed for cleaner code)

    # ---------- FINAL FRAME ---------- #
    frame = ui
    
    # ---------------- COMPACT RESULT BOX ---------------- #

    if show_result and time.time() - result_time > 1.0:
        # Result text format
        r_scale = 0.9
        text_size = cv2.getTextSize(result_text, FONT, r_scale, 2)[0]
        
        # Dynamic Box Dimensions
        pad_x, pad_y = 20, 15
        box_w = text_size[0] + (pad_x * 2)
        box_h = text_size[1] + (pad_y * 2)
        
        x1_box = (w - box_w) // 2
        y1_box = (h - box_h) // 2
        x2_box = x1_box + box_w
        y2_box = y1_box + box_h
        
        # Result text color
        r_color = (100, 255, 100) if "Win" in result_text else (100, 100, 255)
        if "Draw" in result_text or "Invalid" in result_text: r_color = COLOR_TEXT

        # Compact Result Box
        cv2.rectangle(ui, (x1_box, y1_box), (x2_box, y2_box), COLOR_BG, -1)
        # Result Border
        cv2.rectangle(ui, (x1_box, y1_box), (x2_box, y2_box), r_color, 1, cv2.LINE_AA)

        # Center it
        tx = x1_box + pad_x
        ty = y1_box + box_h - pad_y
        cv2.putText(ui, result_text, (tx, ty), FONT, r_scale, r_color, 1, cv2.LINE_AA)
        
    # ---------- SHOW ---------- #
    cv2.imshow("RPS Game", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------- VIRTUAL BUTTON LOGIC ---------- #
    button_hovered = False
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()