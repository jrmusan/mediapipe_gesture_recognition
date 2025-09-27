import time
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Optional, List, Tuple

# === CONFIGURATION ===
# Path to the .task model file. MUST BE placed at the repository/script root.
current_directory = Path(__file__).parent
MODEL_PATH = current_directory / 'gesture_recognizer.task'

# === SHORTCUTS TO THE TASKS API (Just makes things more readable) ===
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

# Shared state for latest gesture (simple string)
_latest_gesture: Optional[str] = None

# Latest landmarks (normalized image coords as floats 0..1)
_latest_landmarks_norm: Optional[List[Tuple[float, float]]] = None
# Standard MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (5,9),(9,10),(10,11),(11,12),   # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20),# Pinky
    (0,17)                          # Palm base connection
]


def print_result(result, output_image, timestamp_ms: int):
    """Callback for live-stream results, because its an asynchronous API we have to use a callback for its reponse. 

    The demo simpily prints the result
    You could also use the landmarks/classification info to drive your logic or draw overlays on the output_image.
    """

    text = None
    if result and result.gestures:
        # The result contains four components
        # Each component is an array
        # where each element contains the detected result of a single detected hand.

        gesture = result.gestures[0][0]
        handedness = result.handedness[0][0].category_name
        text = f"{handedness} hand - {gesture.category_name} - Confidence: ({gesture.score:.2f})"
        print(f'Reconized: {text}')

    # update shared state
    global _latest_gesture, _latest_landmarks_norm
    _latest_gesture = text

    # store normalized image-space landmarks (x, y)
    if result and getattr(result, 'hand_landmarks', None):
        # result.hand_landmarks is a list of lists (hands x landmarks)
        hand0 = result.hand_landmarks[0]
        _latest_landmarks_norm = [(lm.x, lm.y) for lm in hand0]
    else:
        _latest_landmarks_norm = None


def main():
    # Create options for live stream mode.
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, 
        result_callback=print_result, # Used for when recognize_async is called below asynchronously (results come back via this callback defined above) 
    )

    # Create the recognizer
    with GestureRecognizer.create_from_options(options) as recognizer:
        
        # Open the default camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print('Error: could not open camera')
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                 # Flip the frame horizontally (so it acts like a mirror)
                frame_flipped = cv2.flip(frame, 1)   
            
                # MediaPipe expects RGB images. OpenCV gives BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Build an mp.Image from the numpy array. We use SRGB format.
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Timestamp in milliseconds is required for LIVE_STREAM mode.
                timestamp_ms = int(time.time() * 1000)

                # live image data to perform gesture recognition asynchronously
                # results for LIVE_STREAM are delivered via the `print_result` callback
                recognizer.recognize_async(mp_image, timestamp_ms)

                # Draw the latest gesture so you can see what the heck it is
                detected_gesture = _latest_gesture
                if detected_gesture:
                    cv2.putText(frame_flipped, detected_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # ===== Draw hand landmarks, this is totatly unessassry but looks cool as heck =====
                if _latest_landmarks_norm:
                    h, w = frame_flipped.shape[:2]
                    # Convert normalized landmarks to pixel coords.
                    pts = [(int((1.0 - x) * w), int(y * h)) for (x, y) in _latest_landmarks_norm]
                    # draw connections with a dark outline + colored inner line for visibility
                    for a, b in HAND_CONNECTIONS:
                        if a < len(pts) and b < len(pts):
                            # outer outline
                            cv2.line(frame_flipped, pts[a], pts[b], (0, 0, 0), 10, cv2.LINE_AA)
                            # inner colored line
                            cv2.line(frame_flipped, pts[a], pts[b], (0, 255, 0), 6, cv2.LINE_AA)
                    # draw keypoints with outline
                    for (x_px, y_px) in pts:
                        cv2.circle(frame_flipped, (x_px, y_px), 10, (0, 0, 0), -1)
                        cv2.circle(frame_flipped, (x_px, y_px), 6, (0, 0, 255), -1)
                # ===== Draw hand landmarks, this is totatly unessassry but looks cool as heck =====

                # Show the camera feed and stop on 'q'
                cv2.imshow('Gesture Live', frame_flipped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# AUTHOR: Joey Musante - (jrmusan@gmail.com)
