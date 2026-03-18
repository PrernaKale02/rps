# 🎮 Vision-Based Rock Paper Scissors Game

## 📌 Overview

This project implements a **real-time Rock-Paper-Scissors game** using computer vision.
The system uses a webcam to detect hand gestures and allows a user to play against the computer without any physical input devices.

The project demonstrates concepts of:

- Computer Vision
- Hand Gesture Recognition
- Human-Computer Interaction (HCI)
- Real-Time Processing

---

## 🎯 Objective

To develop an interactive system that recognizes hand gestures (Rock, Paper, Scissors) using a camera and enables a user to play the game against a computer in real time.

---

## 🧠 Methodology

1. Capture live video using webcam
2. Detect hand landmarks using MediaPipe
3. Identify finger states (open/closed)
4. Classify gesture as Rock, Paper, or Scissors
5. Generate computer's move randomly
6. Compare moves and determine winner
7. Display result in real time

---

## ⚙️ Technologies Used

- **Python 3**
- **OpenCV** – video capture and display
- **MediaPipe (Tasks API)** – hand landmark detection
- **NumPy** – numerical operations
- **Random module** – generating computer moves

---

## 🏗️ System Architecture

```
Webcam
   ↓
OpenCV (Frame Capture)
   ↓
MediaPipe Hand Landmark Detection
   ↓
Gesture Classification
   ↓
Game Logic (Winner Decision)
   ↓
Display Output (Result on Screen)
```

---

## ✋ Gesture Detection Logic

The system detects gestures by analyzing finger positions:

| Gesture  | Finger State (Index → Pinky) |
| -------- | ---------------------------- |
| Rock     | Closed [0,0,0,0]             |
| Paper    | Open [1,1,1,1]               |
| Scissors | [1,1,0,0]                    |

Thumb is ignored to ensure compatibility with both left and right hands.

---

## 🚀 Features

- Real-time hand tracking
- Works with both left and right hands
- No external hardware required
- Lightweight and fast execution
- Easy to extend with GUI and scoreboard

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

2. Download the model file:

```
hand_landmarker.task
```

3. Place it in the project directory

4. Run the program:

```bash
python main.py
```

5. Press **ESC** to exit

---

## 📊 Output

- Webcam window opens
- Hand landmarks are detected
- Gesture (Rock/Paper/Scissors) is displayed on screen

---

## ⚠️ Limitations

- Performance depends on lighting conditions
- Accuracy reduces with unclear hand gestures
- Only supports three predefined gestures

---

## 🔮 Future Improvements

- Add GUI with scoreboard and rounds
- Integrate text-to-speech for countdown
- Add multiplayer mode
- Improve gesture robustness using ML models

---

## 📚 Applications

- Gesture-controlled interfaces
- Interactive gaming systems
- Touchless user interaction
- Educational demonstrations

---

## 🏁 Conclusion

The project successfully demonstrates a **real-time gesture recognition system** using computer vision techniques. It provides an intuitive and interactive way to play Rock-Paper-Scissors without physical input devices.

---

## 👩‍💻 Author

- Prerna Kale

---
