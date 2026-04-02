import cv2
import mediapipe as mp
import random
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# Load model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ROUND_COUNTDOWN = 3
RESULT_DISPLAY_TIME = 2

result_time = 0
show_result = False
anim_move = "Rock"
anim_phase = 0
last_count = 3

rock_img = cv2.imread("rock.png", cv2.IMREAD_UNCHANGED)
paper_img = cv2.imread("paper.png", cv2.IMREAD_UNCHANGED)
scissors_img = cv2.imread("scissors.png", cv2.IMREAD_UNCHANGED)

if rock_img is None or paper_img is None or scissors_img is None:
    print("❌ ERROR: Images not loaded")
    exit()

player_score = 0
computer_score = 0

player_move = ""
computer_move = ""
result_text = ""

countdown = 0
start_time = None
playing = False


def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def decide_winner(player, computer):
    if player == computer:
        return "Draw 🤝"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Scissors" and computer == "Paper") or \
         (player == "Paper" and computer == "Rock"):
        return "You Win 🎉"
    else:
        return "Computer Wins 🤖"

def detect_rps(landmarks):
    fingers = []
    tips = [8, 12, 16, 20]

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0,0,0,0]:
        return "Rock"
    elif fingers == [1,1,1,1]:
        return "Paper"
    elif fingers == [1,1,0,0]:
        return "Scissors"
    else:
        return "Unknown"

def overlay_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    x = max(0, x)
    y = max(0, y)
    if x >= bg_w or y >= bg_h:
        return bg

    if x + w > bg_w or y + h > bg_h:
        overlay = overlay[:max(0, bg_h - y), :max(0, bg_w - x)]
        h, w = overlay.shape[:2]

    if overlay.shape[2] < 4:
        bg[y:y+h, x:x+w] = overlay
        return bg

    alpha = overlay[:, :, 3:4].astype(float) / 255.0
    alpha_inv = 1.0 - alpha
    bg_region = bg[y:y+h, x:x+w].astype(float)

    blended = alpha * overlay[:, :, :3].astype(float) + alpha_inv * bg_region
    bg[y:y+h, x:x+w] = blended.astype(bg.dtype)

    return bg

def get_computer_image(move):
    if move == "Rock":
        return rock_img
    elif move == "Paper":
        return paper_img
    elif move == "Scissors":
        return scissors_img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    left_x = 40
    right_x = w - 260
    center_x = w // 2
    top_y = 60
    base_x = w - 220
    base_y = 150

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    gesture = "Unknown"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0,255,0), -1)
            gesture = detect_rps(hand_landmarks)


    overlay = frame.copy()

    # left panel
    cv2.rectangle(overlay, (20, 120), (220, 350), (40, 40, 40), -1)

        # right panel
    cv2.rectangle(overlay, (w-220, 120), (w-20, 350), (40, 40, 40), -1)

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # borders
    cv2.rectangle(frame, (20, 120), (220, 350), (0, 255, 100), 2)
    cv2.rectangle(frame, (w-220, 120), (w-20, 350), (255, 200, 0), 2)
    

    if playing:
        elapsed = time.time() - start_time
        countdown = ROUND_COUNTDOWN - int(elapsed)
        
        # detect countdown change (3 → 2 → 1)
        if countdown != last_count:
            anim_phase = 0  # reset animation each step
            last_count = countdown
    
        if countdown == 0:
            cv2.putText(frame, "GO!",(w//2 - 80, h//2),cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 5)
        
        # animation move during countdown
        if countdown > 0:
            anim_move = random.choice(["Rock","Paper","Scissors"])
        
        cv2.putText(frame, "GET READY!",(w//2 - 150, h//2 - 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(frame, "SHOW YOUR MOVE",(w//2 - 180, h//2 - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        if countdown > 0:
            scale = 3 + (0.5 * (ROUND_COUNTDOWN - countdown))  # zoom effect

            text = str(countdown)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 5)[0]
            x = (w - text_size[0]) // 2
            y = (h + text_size[1]) // 2

            cv2.putText(frame, text, (x, y),cv2.FONT_HERSHEY_SIMPLEX, scale,(0, 0, 255), 5)
            
        else:
            if not show_result:
                if gesture != "Unknown":
                    player_move = gesture
                    computer_move = get_computer_choice()
                    result_text = decide_winner(player_move, computer_move)

                    if result_text == "You Win 🎉":
                        player_score += 1
                    elif result_text == "Computer Wins 🤖":
                        computer_score += 1
                else:
                    player_move = ""
                    computer_move = ""
                    result_text = "Invalid Gesture!"

                show_result = True
                result_time = time.time()

            playing = False

    
    if show_result:
        # shadow
        cv2.rectangle(frame,
                  (w//2 - 220, h//2 - 60),
                  (w//2 + 220, h//2 + 60),
                  (0, 0, 0), -1)

        # main card
        cv2.rectangle(frame,
                  (w//2 - 200, h//2 - 50),
                  (w//2 + 200, h//2 + 50),
                  (50, 50, 50), -1)

        cv2.rectangle(frame,
                  (w//2 - 200, h//2 - 50),
                  (w//2 + 200, h//2 + 50),
                  (255, 0, 255), 3)

        # centered text
        text_size = cv2.getTextSize(result_text,cv2.FONT_HERSHEY_SIMPLEX,1, 3)[0]
        x = (w - text_size[0]) // 2

        cv2.putText(frame, result_text,(x, h//2 + 10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 3)
        if time.time() - result_time >= RESULT_DISPLAY_TIME:
            show_result = False
            player_move = ""
            computer_move = ""
            result_text = ""
                
    # YOU
    cv2.putText(frame, "YOU", (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

    # COMPUTER
    cv2.putText(frame, "COMPUTER", (w-260, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)

    # SCORE
    score_text = f"{player_score} : {computer_score}"
    text_size = cv2.getTextSize(score_text,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]

    x = (w - text_size[0]) // 2

    cv2.putText(frame, score_text, (x, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

    # glow effect
    cv2.putText(frame, score_text, (x, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)

    cv2.putText(frame, score_text, (x, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

    

    cv2.putText(frame, "Press SPACE to Play | R to Reset | ESC/Q to Exit",(w//2 - 280, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200, 200, 200), 2)


    player_base_x = 40
    player_base_y = 150
    computer_base_x = w - 220
    computer_base_y = 150

    if playing and countdown > 0:
        comp_img = get_computer_image(anim_move) # animation (rock)
        anim_phase += 0.4
        offset_y = int(20 * abs(math.sin(anim_phase)))  # bounce up-down
        offset_x = int(5 * math.sin(anim_phase))
    elif computer_move:
        comp_img = get_computer_image(computer_move)  # final move
        offset_x = 0
        offset_y = 0
    else:
        comp_img = None
        offset_x = 0
        offset_y = 0

    player_img = get_computer_image(player_move) if player_move else None

    if player_img is not None:
        player_img = cv2.resize(player_img, (200, 200))
        frame = overlay_image(frame, player_img, player_base_x, player_base_y)

    if comp_img is not None:
        comp_img = cv2.resize(comp_img, (200, 200))
        frame = overlay_image(frame, comp_img, computer_base_x + offset_x, computer_base_y + offset_y)
            
    cv2.imshow("RPS Pro Game", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32 and not playing:  # SPACE
        playing = True
        start_time = time.time()
        show_result = False   #  RESET THIS
        player_move = ""
        computer_move = ""
        result_text = ""

    if key == ord('r'):
        player_score = 0
        computer_score = 0
        result_text = ""
        show_result = False

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()