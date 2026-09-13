# src/haar_5pt.py
"""
Haar face detection + practical 5-point landmarks (MediaPipe FaceLandmarker Tasks API).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path("models/face_landmarker.task")


@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32


def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    k = kps_5x2.astype(np.float32)

    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    out_w, out_h = int(out_size[0]), int(out_size[1])

    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )

    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> Tuple[np.ndarray, np.ndarray]:
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])

    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _clip_box_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = b.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k

    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < min_eye_dist:
        return False

    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False

    return True


class Haar5ptDetector:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

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
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._ts_ms = 0

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts_ms += 1
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        idxs = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)

        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)

        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = faces[i].tolist()

        kps = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[haar_5pt] Haar face found but FaceLandmarker returned none -> reject")
            return []

        margin = 0.35
        x1m = x - margin * w
        y1m = y - margin * h
        x2m = x + (1.0 + margin) * w
        y2m = y + (1.0 + margin) * h

        inside = (
            (kps[:, 0] >= x1m) & (kps[:, 0] <= x2m) &
            (kps[:, 1] >= y1m) & (kps[:, 1] <= y2m)
        )
        if inside.mean() < 0.60:
            if self.debug:
                print("[haar_5pt] FaceLandmarker points not consistent with Haar box -> reject")
            return []

        if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
            if self.debug:
                print("[haar_5pt] 5pt geometry sanity failed -> reject")
            return []

        box = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
        box = _clip_box_xyxy(box, W, H)

        box_s = _ema(self._prev_box, box, self.smooth_alpha)
        kps_s = _ema(self._prev_kps, kps, self.smooth_alpha)

        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        x1, y1, x2, y2 = box_s.tolist()
        score = 1.0

        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=float(score),
                kps=kps_s.astype(np.float32),
            )
        ][:max_faces]


def main():
    cap = cv2.VideoCapture(0)
    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=True)

    print("Haar + 5pt (FaceLandmarker) test. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=1)
        vis = frame.copy()

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(vis, "OK", (f.x1, max(0, f.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("haar_5pt", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()