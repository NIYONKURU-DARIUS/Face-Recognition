# src/face_tracking.py
import argparse
import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.align import align_face_5pt
from src.face_signals import FaceSignalExtractor
from src.haar_5pt import Haar5ptDetector          # <-- was "HaarFaceMesh5pt" from src.recognize
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return match.accepted and match.name == self.target_name

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None

        last_center = center(self.last_box)
        last_diag = max(np.linalg.norm(
            np.array([self.last_box[2] - self.last_box[0],
                      self.last_box[3] - self.last_box[1]], dtype=np.float32)
        ), 1.0)

        ranked = []
        for face in faces:
            box = self.box(face)
            overlap = iou(self.last_box, box)
            displacement = np.linalg.norm(center(box) - last_center) / last_diag
            score = overlap - 0.35 * displacement
            ranked.append((score, face))

        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = self.detector.detect(frame, max_faces=8)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        if (
            candidate is not None
            and (self.state == LockState.LOST
                 or self.frame_index % self.verify_every == 0)
            and not self.target_is_verified(frame, candidate)
        ):
            candidate = None

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)

        raw_center = center(self.last_box)
        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center

        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))

        horizontal = "CENTER"
        vertical = "CENTER"
        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"
        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"

        return TrackingSignal(ex, ey, horizontal, vertical)


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def signal_dict(state: str, target: str, locked: bool, frame_n: int, position=None, face_state=None, blink_total: int = 0):
    d = {
        "part": 2,
        "frame": frame_n,
        "lock_state": state,
        "target": target,
        "identity_locked": locked,
    }
    if locked and position is not None:
        d.update(
            error_x=round(position.error_x, 4),
            error_y=round(position.error_y, 4),
            horizontal=position.horizontal,
            vertical=position.vertical,
        )
    if locked and face_state is not None:
        d.update(
            blink=bool(face_state.blink),
            eyes_closed=bool(face_state.eyes_closed),
            ear=round(face_state.ear, 4),
            smiling=bool(face_state.smiling),
            smile_score=round(face_state.smile_score, 4),
            blink_total=blink_total,
        )
    return d


def main():
    parser = argparse.ArgumentParser(description="Part 2 - Face tracking with identity lock")
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.34, help="ArcFace distance threshold (default: 0.34)")
    parser.add_argument("--ear-threshold", type=float, default=0.23, help="EAR threshold for eye closure (default: 0.23)")
    parser.add_argument("--smile-on", type=float, default=0.39, help="Smile ratio threshold to activate SMILE (default: 0.39)")
    parser.add_argument("--smile-off", type=float, default=0.36, help="Smile ratio threshold to exit SMILE (default: 0.36)")
    parser.add_argument("--blink-min-frames", type=int, default=1, help="Min consecutive frames below EAR for blink (default: 1)")
    parser.add_argument("--signal", action="store_true", help="headless: print one JSON signal line per frame (Part 3 feed)")
    parser.add_argument("--max-frames", type=int, default=0, help="quit after N frames in --signal mode (0 = never)")
    parser.add_argument("--width", type=int, default=0, help="requested camera width (0 = driver default)")
    parser.add_argument("--height", type=int, default=0, help="requested camera height (0 = driver default)")
    args = parser.parse_args()

    # max_faces=8 is the load-bearing change vs. the Part 1 detector: the
    # identity lock and the distractor test both need to see more than one
    # face at a time.
    detector = Haar5ptDetector(min_size=(70, 70), debug=False, max_faces=8)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )

    tracker = LockedFaceTracker(args.target, detector, embedder, matcher)
    signals = FaceSignalExtractor(
        ear_threshold=args.ear_threshold,
        blink_min_frames=args.blink_min_frames,
        smile_on=args.smile_on,
        smile_off=args.smile_off,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    if args.width > 0 and args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not args.signal:
        cv2.namedWindow("Locked Face Tracking", cv2.WINDOW_NORMAL)

    blink_total = 0
    signal_miss_count = 0
    last_face_state = None
    calibrated_msg_timer = 0
    frame_n = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1

            locked_face, position = tracker.update(frame)
            view = frame.copy() if not args.signal else None

            face_state = None
            if locked_face is not None:
                box = tracker.box(locked_face)
                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    signal_miss_count = 0
                    last_face_state = face_state
                    if face_state.blink:
                        blink_total += 1
                else:
                    signal_miss_count += 1
                    # Only reset if landmarks were missed for several consecutive frames
                    if signal_miss_count > 4:
                        signals.reset()
            else:
                signals.reset()
                signal_miss_count = 0

            if args.signal:
                out = signal_dict(tracker.state.name, args.target,
                                  locked_face is not None, frame_n,
                                  position=position, face_state=face_state,
                                  blink_total=blink_total)
                print(json.dumps(out, sort_keys=True), flush=True)
                if args.max_frames and frame_n >= args.max_frames:
                    break
            else:
                state_text = f"{tracker.state.name}: {args.target}"
                state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
                draw_label(view, state_text, (12, 28), state_color, 0.72)

                if locked_face is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                    if face_state is not None:
                        expression = "SMILE" if face_state.smiling else "NEUTRAL"
                        eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                        draw_label(view, expression, (x1, max(55, y1 - 50)), (0, 255, 255))
                        draw_label(view, f"{eye_text} blinks={blink_total}",
                                   (x1, max(78, y1 - 25)), (255, 255, 0))
                        draw_label(view, f"EAR={face_state.ear:.3f} (th={signals.ear_threshold:.2f}) "
                                          f"smile={face_state.smile_score:.3f} (on={signals.smile_on:.2f}, off={signals.smile_off:.2f})",
                                   (12, view.shape[0] - 18), (255, 255, 255), 0.52)

                    draw_label(view,
                               f"H={position.horizontal} V={position.vertical} "
                               f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                               (12, 56), (255, 170, 0), 0.60)

                # UI hints and calibration notification
                if calibrated_msg_timer > 0:
                    draw_label(view, "Calibrated neutral baseline!", (12, 85), (0, 255, 0), 0.65)
                    calibrated_msg_timer -= 1
                else:
                    draw_label(view, "[Press 'c': Calibrate neutral | 'q': Quit]", (12, 85), (200, 200, 200), 0.50)

                h, w = view.shape[:2]
                dz = tracker.dead_zone
                cv2.rectangle(view,
                              (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                              (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                              (120, 120, 120), 1)

                cv2.imshow("Locked Face Tracking", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c") and last_face_state is not None:
                    signals.calibrate(last_face_state.ear, last_face_state.smile_score)
                    calibrated_msg_timer = 30
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()