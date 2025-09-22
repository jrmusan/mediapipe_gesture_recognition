import cv2
import mediapipe as mp
import numpy as np

from samsungtvws import SamsungTVWS

ip = "192.168.12.222"
token_file = ".samsungtv.token"


tv = SamsungTVWS(ip, port=8002, token_file=token_file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

    
def is_open_palm(hand_landmarks):
    """Detect if the hand is open (all fingers extended)."""
    # Check if fingertips are above their PIP joints (y is smaller higher up)
    open_fingers = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:  # index, middle, ring, pinky
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            open_fingers += 1
    return open_fingers == 4  # all four fingers extended


def is_fist(hand_landmarks):
    """Detect if the hand is closed (all fingers folded)."""
    folded_fingers = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    return folded_fingers == 4  # all four folded

    
def classify_pointing(hand_landmarks):
    """
    Classify a pointing gesture (left or right) based on MediaPipe hand landmarks.
    Returns:
        "Point Left", "Point Right", or None
    """
    # Index finger tip and pip
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    wrist = hand_landmarks.landmark[0]

    # 1. Is index finger extended? (tip higher than pip)
    index_extended = index_tip.y < index_pip.y

    # 2. Are other fingers folded?
    folded = True
    for tip, pip in [(12,10), (16,14), (20,18)]:  # middle, ring, pinky
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            folded = False

    if index_extended and folded:
        # 3. Direction: compare index tip x to wrist x
        if index_tip.x < wrist.x:
            tv.shortcuts().left()
            return "Point Left"
        else:
            tv.shortcuts().right()
            return "Point Right"
    elif is_open_palm(hand_landmarks):
        tv.shortcuts().enter()
        return "Open Palm"
    elif is_fist(hand_landmarks):
        tv.shortcuts().home()
        return "fist"

    return None

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and process
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_pointing(hand_landmarks)
                cv2.putText(image, gesture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Remote", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
