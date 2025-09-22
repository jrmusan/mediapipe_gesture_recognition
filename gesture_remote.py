import cv2
import mediapipe as mp
import numpy as np

from samsungtvws import SamsungTVWS

ip = "192.168.12.222"
token_file = ".samsungtv.token"


tv = SamsungTVWS(ip, port=8002, token_file=token_file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Helper: check if fingers are extended
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    pips = [3, 6, 10, 14, 18]  # corresponding lower joints
    
    finger_status = []
    for tip, pip in zip(tips[1:], pips[1:]):  # skip thumb for now
        finger_status.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    
    # Simple thumb check (x-axis, since it points sideways)
    thumb_extended = hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[pips[0]].x
    finger_status.insert(0, thumb_extended)
    return finger_status

# Simple classifier: palm vs fist
def classify_gesture(hand_landmarks):
    finger_status = fingers_up(hand_landmarks)
    if sum(finger_status) == 0:
        tv.shortcuts().volume_up()
        return "Fist"
    elif sum(finger_status) >= 4:
        tv.shortcuts().volume_down()
        return "Open Palm"
    else:
        return "Other"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip and process
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks)
                cv2.putText(image, gesture, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Remote", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
