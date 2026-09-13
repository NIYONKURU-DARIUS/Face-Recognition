"""
Falcon Eye - Face Recognition + Servo Scanner (automatic, center-out, hold-and-check)

Pipeline:

Camera
   -> Haar face detection
   -> MediaPipe FaceLandmarker 5-point landmarks
   -> 5-point alignment
   -> ArcFace ONNX embedding
   -> Face database
   -> Known / Stranger
   -> MQTT
   -> ESP32
   -> servo

Behavior:

- Servo starts at 90 degrees (center).
- Scanning starts automatically the moment the program is ready -- no
  keypress needed.
- The servo only moves to a new position if the known face was NOT found
  after holding at the current position for POSITION_HOLD_TIME seconds.
- At each position, the camera checks periodically for the full hold time.
- If the known face is found at any point -> STOP, hold that position,
  and keep watching there (does not resume scanning while you remain).
- If a face is found but it does not match the database -> "Stranger"
  (logged + published over MQTT), scanning continues at the same position
  until the hold time runs out.
- If all scan positions are exhausted with no match -> NOT_FOUND, then
  the whole sweep restarts automatically from 90 degrees and keeps trying.
- Press 'q' at any time to quit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt

from .haar_5pt import align_face_5pt, Haar5ptDetector


# ============================================================
# PATHS
# ============================================================

DB_PATH = Path("data/db/face_db.npz")

ARC_FACE_MODEL = "models/embedder_arcface.onnx"


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT_HOST = "broker.benax.rw"
MQTT_PORT = 1883

TOPIC_SERVO_CMD = "falcon/eye/servo/cmd"
TOPIC_SERVO_STATUS = "falcon/eye/servo/status"
TOPIC_RECOGNITION = "falcon/eye/recognition"


# ============================================================
# SCANNER CONFIGURATION
# ============================================================

# Start centered, then sweep outward alternating right/left
SCAN_ANGLES = [90, 110, 70, 130, 50, 150, 30, 170, 10, 180, 0]

NUMBER_OF_SCANS = len(SCAN_ANGLES)

HOME_ANGLE = 90

# Time for servo to move/stabilize before checking faces
SERVO_SETTLE_TIME = 0.8

# How long to stay at each position looking for the known face
POSITION_HOLD_TIME = 5.0

# How often to sample a frame while holding at a position
FRAME_CHECK_INTERVAL = 0.5

# ArcFace acceptance threshold
DISTANCE_THRESHOLD = 0.34


# ============================================================
# MATCH RESULT
# ============================================================

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# ============================================================
# COSINE FUNCTIONS
# ============================================================

def cosine_similarity(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)


# ============================================================
# FACE DATABASE
# ============================================================

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        print("[DB] Database does not exist:", db_path)
        return {}

    data = np.load(str(db_path), allow_pickle=True)
    out = {}

    for key in data.files:
        out[key] = np.asarray(data[key], dtype=np.float32).reshape(-1)

    return out


# ============================================================
# ARC FACE
# ============================================================

class ArcFaceEmbedderONNX:
    def __init__(self, model_path=ARC_FACE_MODEL, input_size=(112, 112)):
        self.model_path = model_path
        self.in_w = int(input_size[0])
        self.in_h = int(input_size[1])

        print("[ArcFace] Loading:", model_path)

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        print("[ArcFace] Input:", self.in_name)
        print("[ArcFace] Output:", self.out_name)

    def _preprocess(self, aligned_bgr):
        img = aligned_bgr

        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]

        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(vector, eps=1e-12):
        vector = vector.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector) + eps)
        return (vector / norm).astype(np.float32)

    def embed(self, aligned_bgr):
        x = self._preprocess(aligned_bgr)
        output = self.sess.run([self.out_name], {self.in_name: x})[0]
        embedding = np.asarray(output, dtype=np.float32).reshape(-1)
        return self._l2_normalize(embedding)


# ============================================================
# FACE DATABASE MATCHER
# ============================================================

class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh=DISTANCE_THRESHOLD):
        self.db = db
        self.dist_thresh = float(dist_thresh)

        self._names = []
        self._mat = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())

        if self._names:
            self._mat = np.stack(
                [self.db[name].reshape(-1).astype(np.float32) for name in self._names],
                axis=0
            )
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, embedding) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = embedding.reshape(1, -1).astype(np.float32)
        similarities = (self._mat @ e.T).reshape(-1)

        best_i = int(np.argmax(similarities))
        best_similarity = float(similarities[best_i])
        best_distance = 1.0 - best_similarity

        accepted = best_distance <= self.dist_thresh

        return MatchResult(
            name=self._names[best_i] if accepted else None,
            distance=best_distance,
            similarity=best_similarity,
            accepted=accepted
        )


# ============================================================
# MQTT SERVO CONTROLLER
# ============================================================

class ServoController:
    def __init__(self):
        self.connected = False
        self.last_status = None

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        print("[MQTT] Connecting to:", MQTT_HOST)

        self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()

        timeout = time.time() + 5
        while not self.connected and time.time() < timeout:
            time.sleep(0.05)

        if not self.connected:
            raise RuntimeError("Could not connect to MQTT broker")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.connected = True
            print("[MQTT] Connected")
            client.subscribe(TOPIC_SERVO_STATUS)
        else:
            print("[MQTT] Connection failed:", reason_code)

    def _on_message(self, client, userdata, message):
        try:
            payload = message.payload.decode()
            self.last_status = payload
            print("[ESP]", payload)
        except Exception as e:
            print("[MQTT] Message error:", e)

    def move_to(self, angle: int):
        angle = max(0, min(180, int(angle)))
        command = f"ANGLE:{angle}"
        print("[MQTT] -> ESP:", command)
        self.client.publish(TOPIC_SERVO_CMD, command)

    def stop(self):
        print("[MQTT] -> ESP: STOP")
        self.client.publish(TOPIC_SERVO_CMD, "STOP")

    def home(self):
        print("[MQTT] -> ESP: HOME")
        self.client.publish(TOPIC_SERVO_CMD, "HOME")

    def publish_recognition(self, name, distance, similarity, angle, attempt):
        payload = {
            "name": name,
            "distance": float(distance),
            "similarity": float(similarity),
            "angle": int(angle),
            "attempt": int(attempt)
        }
        self.client.publish(TOPIC_RECOGNITION, json.dumps(payload))

    def publish_not_found(self):
        payload = {
            "name": None,
            "status": "NOT_FOUND",
            "attempts": NUMBER_OF_SCANS
        }
        self.client.publish(TOPIC_RECOGNITION, json.dumps(payload))

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_frame(frame, detector, embedder, matcher):
    faces = detector.detect(frame, max_faces=5)

    if not faces:
        return None, faces

    best_result = None

    for face in faces:
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        embedding = embedder.embed(aligned)
        result = matcher.match(embedding)

        if best_result is None or result.distance < best_result.distance:
            best_result = result

    return best_result, faces


# ============================================================
# DRAWING
# ============================================================

def draw_faces(frame, faces):
    vis = frame.copy()

    for face in faces:
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)

        for x, y in face.kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    return vis


# ============================================================
# ONE FULL SWEEP (returns name if found, None if exhausted)
# Also handles the display window + 'q' quit check throughout.
# ============================================================

def run_one_sweep(cap, detector, embedder, matcher, servo) -> Tuple[Optional[str], bool]:
    """
    Returns (found_name_or_None, quit_requested)
    """

    print()
    print("=" * 60)
    print("STARTING FACE SCAN (start: 90 degrees)")
    print("=" * 60)
    print("Scan positions:", SCAN_ANGLES)

    for attempt, angle in enumerate(SCAN_ANGLES, start=1):

        print()
        print(f"[SCAN {attempt}/{NUMBER_OF_SCANS}] Moving to {angle} degrees")

        servo.move_to(angle)
        time.sleep(SERVO_SETTLE_TIME)

        position_start = time.time()
        stranger_announced = False

        while (time.time() - position_start) < POSITION_HOLD_TIME:

            ok, frame = cap.read()

            if not ok:
                print("[CAMERA] Failed to read frame")
                time.sleep(FRAME_CHECK_INTERVAL)
                continue

            result, faces = recognize_frame(frame, detector, embedder, matcher)

            # live preview window
            vis = draw_faces(frame, faces)
            header = f"scanning {angle} deg | IDs={len(matcher._names)} thr={matcher.dist_thresh:.2f}"
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Falcon Eye", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None, True

            # ----------------------------------------------
            # NO FACE IN FRAME
            # ----------------------------------------------
            if result is None:
                time.sleep(FRAME_CHECK_INTERVAL)
                continue

            # ----------------------------------------------
            # KNOWN FACE -> STOP IMMEDIATELY
            # ----------------------------------------------
            if result.accepted:

                print()
                print("=" * 60)
                print("KNOWN FACE FOUND!")
                print("Name:", result.name)
                print("Distance:", f"{result.distance:.3f}")
                print("Similarity:", f"{result.similarity:.3f}")
                print("Angle:", angle)
                print("=" * 60)

                servo.stop()

                servo.publish_recognition(
                    name=result.name,
                    distance=result.distance,
                    similarity=result.similarity,
                    angle=angle,
                    attempt=attempt
                )

                return result.name, False

            # ----------------------------------------------
            # UNKNOWN FACE -> STRANGER
            # ----------------------------------------------
            else:
                if not stranger_announced:
                    print(
                        f"[SCAN {attempt}] Stranger detected at {angle} degrees "
                        f"(dist={result.distance:.3f})"
                    )

                    servo.publish_recognition(
                        name="Stranger",
                        distance=result.distance,
                        similarity=result.similarity,
                        angle=angle,
                        attempt=attempt
                    )

                    stranger_announced = True

                time.sleep(FRAME_CHECK_INTERVAL)

        print(
            f"[SCAN {attempt}] No known match at {angle} degrees "
            f"after {POSITION_HOLD_TIME:.0f}s, moving on"
        )

    # all positions exhausted, nobody found this sweep
    servo.publish_not_found()

    print()
    print("=" * 60)
    print("NOT FOUND (this sweep) -- restarting scan")
    print(f"Completed {NUMBER_OF_SCANS} scan positions")
    print("=" * 60)

    return None, False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FALCON EYE FACE RECOGNITION (automatic mode)")
    print("=" * 60)

    # --------------------------------------------------------
    # FACE DETECTOR
    # --------------------------------------------------------
    detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    # --------------------------------------------------------
    # ARC FACE
    # --------------------------------------------------------
    embedder = ArcFaceEmbedderONNX(model_path=ARC_FACE_MODEL, input_size=(112, 112))

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------
    db = load_db_npz(DB_PATH)
    matcher = FaceDBMatcher(db=db, dist_thresh=DISTANCE_THRESHOLD)

    print("[DB] Identities:", len(matcher._names))

    if matcher._names:
        print("[DB] Names:", ", ".join(matcher._names))
    else:
        print("[WARNING] Face database is empty!")

    # --------------------------------------------------------
    # MQTT / SERVO
    # --------------------------------------------------------
    servo = ServoController()

    # Start centered
    servo.move_to(HOME_ANGLE)
    time.sleep(0.5)

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        servo.close()
        raise RuntimeError("Camera not available")

    print()
    print("Camera ready")
    print("MQTT ready")
    print()
    print("Scanning starts automatically. Press 'q' in the video window to quit.")

    # --------------------------------------------------------
    # AUTOMATIC LOOP: keep sweeping until found, then keep
    # watching that spot; if they leave, sweeping resumes.
    # --------------------------------------------------------
    try:
        while True:

            found_name, quit_requested = run_one_sweep(cap, detector, embedder, matcher, servo)

            if quit_requested:
                break

            if found_name:
                print(f"RESULT: {found_name} -- holding position and watching")

                # Stay here and keep confirming the known face is still present.
                # If they leave (face not seen for a while), resume scanning.
                missed_checks = 0
                MAX_MISSED_CHECKS = 6  # ~3s of no detection before resuming scan

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(FRAME_CHECK_INTERVAL)
                        continue

                    result, faces = recognize_frame(frame, detector, embedder, matcher)

                    vis = draw_faces(frame, faces)
                    cv2.putText(vis, f"Watching: {found_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Falcon Eye", vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        quit_requested = True
                        break

                    if result is not None and result.accepted and result.name == found_name:
                        missed_checks = 0
                    else:
                        missed_checks += 1

                    if missed_checks >= MAX_MISSED_CHECKS:
                        print(f"[WATCH] {found_name} no longer detected -- resuming scan")
                        break

                    time.sleep(FRAME_CHECK_INTERVAL)

                if quit_requested:
                    break

            # loop back and sweep again automatically (whether NOT_FOUND
            # or the watched person left)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo.close()

    print("Falcon Eye stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()