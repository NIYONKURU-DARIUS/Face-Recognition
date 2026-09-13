# src/landmarks.py
"""
camera -> Haar face box -> MediaPipe FaceLandmarker (Tasks API) -> extract 5 keypoints -> draw

Uses the newer MediaPipe Tasks API (FaceLandmarker) since the legacy
mp.solutions.face_mesh API has been removed from current mediapipe releases.
Requires models/face_landmarker.task (see download command in the guide).
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

MODEL_PATH = Path("models/face_landmarker.task")


def _make_landmarker() -> mp_vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            "Download it with:\n"
            '  Invoke-WebRequest -Uri '
            '"https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task" '
            '-OutFile "models\\face_landmarker.task"'
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    if face.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    landmarker = _make_landmarker()

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("Haar + FaceLandmarker 5pt (minimal). Press 'q' to quit.")

    frame_timestamp_ms = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 1
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]

            pts = []
            for i in idxs:
                p = lm[i]
                pts.append([p.x * W, p.y * H])
            kps = np.array(pts, dtype=np.float32)

            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            cv2.putText(frame, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("5pt Landmarks", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()