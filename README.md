# Gesture Recognizer Demo
I recently got into AI and ML and had trouble finding good recent examples of using MediaPipe to recognize
Hand gestures. Even [Googles official docs](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#live-stream) wasn't very clear in my opinion. docs. So alas, after a bit of research I have a straightforward example of using Google's MediaPipe that *hopefully* anyone can follow. 


This repository contains a minimal example that demonstrates using MediaPipe's
GestureRecognizer in live-stream mode (camera input). The `demo.py` script
opens your webcam, feeds frames into the recognizer, and prints gesture
classification results via a callback.

## Quickstart

1. Install Python 3.8+ and create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download the MediaPipe gesture recognizer [.task file model here](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models) and place it into the root of the repository.
   

4. Run the demo:

```bash
python demo.py
```
NOTE: It takes like ~20 secs the first time to start running

Press `q` in the camera window to exit.

## Files
- `demo.py` — Live camera demo using MediaPipe GestureRecognizer in
  LIVE_STREAM mode.
- `gesture_recognizer.task` — (not included) The model file expected by the
  demo. Obtain a model from MediaPipe or export one compatible with MediaPipe
  Tasks.
- `requirements.txt` — Python dependencies used by this project.


## License
mediapipe_gesture_recognition is under [Apache v2 license](LICENSE).