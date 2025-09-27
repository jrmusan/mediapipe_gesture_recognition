import time
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Optional

# NOTE: this file is a minimal live-stream example following the
# Google MediaPipe GestureRecognizer demo. 

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


def print_result(result, output_image, timestamp_ms: int):
    """Callback for live-stream results, because its an asynchronous API we have to use a callback for its reponse. 

    The demo simpily prints the result
    You could also use the result you'd use the landmarks/classification info to drive your logic or draw overlays on the output_image.
    """
    text = None
    if result and result.gestures:
        g = result.gestures[0][0]  # first gesture, first category
        # include name and optionally score
        text = f"{g.category_name} Confidence: ({g.score:.2f})"
        print(f'gesture recognition result {text}')

    # update shared state
    global _latest_gesture
    _latest_gesture = text


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

                # MediaPipe expects RGB images. OpenCV gives BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Build an mp.Image from the numpy array. We use SRGB format.
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Timestamp in milliseconds is required for LIVE_STREAM mode.
                timestamp_ms = int(time.time() * 1000)

                # live image data to perform gesture recognition asynchronously
                # results for LIVE_STREAM are delivered via the `print_result` callback
                recognizer.recognize_async(mp_image, timestamp_ms)

                # ~~~~~~~ ALL THIS NONESNSE BELOW IS JUST TO SHOW THE GESTURE ON THE CAMERA FEED ~~~~~~~
                label = _latest_gesture
                if label:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    pad = 8
                    x, y = 10, 30  # top-left origin for label
                    # background rectangle (black, filled)
                    cv2.rectangle(frame, (x - pad//2, y - h - pad//2), (x + w + pad, y + pad//2), (0, 0, 0), -1)
                    # text (white)
                    cv2.putText(frame, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                # ~~~~~~~ ALL THIS NONESNSE ABOVE IS JUST TO SHOW THE GESTURE ON THE CAMERA FEED ~~~~~~~

                # Show the camera feed and stop on 'q'
                cv2.imshow('Gesture Live', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# AUTHOR: Joey Musante - (jrmusan@gmail.com)