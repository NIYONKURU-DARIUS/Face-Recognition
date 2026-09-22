"""
Falcon Eye - Face Recognition + Servo Tracker (full-range sweep, direction-following)

Pipeline:

Camera
   -> Haar face detection
   -> MediaPipe FaceLandmarker 5-point landmarks
   -> 5-point alignment
   -> ArcFace ONNX embedding
   -> Face database
   -> Target / Known-not-target / Stranger
   -> MQTT
   -> ESP32
   -> servo

Behavior:

- Servo starts at HOME_ANGLE (90 degrees).
- Among the known identities in the database, you pick ONE "target" to
  follow (TARGET_NAME below, or you'll be prompted at startup).
- A full search is a continuous sweep across the WHOLE range, in two legs:
      Leg 1: current angle -> 0 degrees
      Leg 2: 0 degrees      -> 180 degrees
  The servo moves in small continuous steps (not big jumps + long holds),
  checking the camera at every step. Only if BOTH legs complete with no
  sighting is the target's absence "confirmed" (logged + published over
  MQTT), and the whole sweep restarts from HOME_ANGLE.
- If the target is found at any point during a leg, the servo STOPS
  sweeping and LOCKS on: it actively tracks/follows the target's
  left-right movement in frame in real time.
- When the target then disappears from view, Falcon Eye does NOT restart
  the whole search. It simply resumes the continuous sweep in the exact
  same direction it was already heading (toward that leg's end angle)
  from wherever the servo currently is. If it reaches that leg's end
  angle without finding them again, that direction is confirmed empty
  and it moves on to the next leg. Only once every direction has been
  covered without a sighting does it conclude "not there" (absence
  confirmed).
- A known face that is NOT the chosen target is reported as "known, not
  target" and does not cause a lock -- the sweep continues.
- Any unrecognized face is reported as "Stranger" (console + MQTT + a
  vivid red on-screen banner).
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
# WHO TO FOLLOW
# ============================================================

# Set this to a name that exists in your face database to skip the
# startup prompt (e.g. TARGET_NAME = "Darius"). Leave as None to be
# prompted with a list of known identities every run.
TARGET_NAME: Optional[str] = None


# ============================================================
# SERVO / SWEEP CONFIGURATION
# ============================================================

HOME_ANGLE = 90

# Time to let the servo settle after a large jump (e.g. moving to the
# start of a new leg).
SERVO_SETTLE_TIME = 0.8

# Degrees per step while continuously sweeping, and the pause after each
# small step to let the servo settle before the camera check. Smaller
# step / longer pause = more thorough but slower; tune for your rig.
SCAN_STEP_ANGLE = 5
SCAN_STEP_SETTLE_TIME = 3

# ArcFace acceptance threshold
DISTANCE_THRESHOLD = 0.5


# ============================================================
# TRACKING CONFIGURATION (following the target while locked on)
# ============================================================

# Flip to -1 if the servo turns the "wrong way" relative to how the
# target appears to move in frame (calibrate once for your rig).
# Confirmed -1 for this rig: with +1, correction nudges kept moving the
# servo monotonically AWAY from a stationary target (offset never
# shrank -- it hit TRACK_MAX_STEP every cycle until the target fell out
# of frame), which is the signature of an inverted feedback loop.
SERVO_DIRECTION_SIGN = -1

# Degrees of servo movement per full-frame horizontal offset (offset
# ranges from -1.0 at the left edge to +1.0 at the right edge).
# Lowered from 25.0: that gain, combined with TRACK_MAX_STEP, was swinging
# the servo past a stationary face on every correction, which is what was
# causing the found -> lost -> found cycling.
TRACK_GAIN = 10.0

# Max degrees the servo is allowed to move in a single tracking nudge,
# to keep motion smooth instead of jerky.
TRACK_MAX_STEP = 3.0

# Ignore small offsets near center so the servo doesn't hunt/jitter.
TRACK_DEADZONE = 0.12

# How often to sample a frame while locked on and tracking (when no
# nudge was made this cycle -- i.e. you're already centered).
FRAME_CHECK_INTERVAL = 0.5

# Extra settle time to wait after a nudge before grabbing the next frame.
# Without this, the camera can still be mid-turn (motion blur / you
# temporarily out of frame) when the next frame is captured, which reads
# as a "missed" detection even though you never left.
TRACK_SETTLE_TIME = 0.5

# How many consecutive missed checks before we consider the target
# actually gone (rather than a momentary blink/occlusion/settle frame).
MISSED_CHECKS_BEFORE_LOST = 8


# ============================================================
# CHASE CONFIGURATION (recovering a target that just left frame)
# ============================================================

# When the target disappears, before falling back to the normal sweep we
# make a quick, more aggressive push further in the direction they were
# last drifting -- i.e. the edge of frame they exited from -- since
# that's the most likely direction they actually walked.
CHASE_STEP_ANGLE = 8
CHASE_STEP_SETTLE_TIME = 0.35

# How far past the angle where we lost them we're willing to chase
# before giving up and declaring them genuinely lost.
CHASE_MAX_DEGREES = 45


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
        self.current_angle = HOME_ANGLE  # last angle WE commanded
        self.actual_angle = HOME_ANGLE   # last angle the ESP32 confirmed reaching

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

            # Keep our notion of the PHYSICAL angle in sync with what the
            # ESP32 actually confirms, rather than trusting the optimistic
            # angle we set in move_to() the instant we send a command.
            try:
                data = json.loads(payload)
                if data.get("status") == "ANGLE_REACHED" and "angle" in data:
                    self.actual_angle = int(data["angle"])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        except Exception as e:
            print("[MQTT] Message error:", e)

    def move_to(self, angle: float):
        angle = max(0, min(180, int(round(angle))))
        self.current_angle = angle
        command = f"ANGLE:{angle}"
        self.client.publish(TOPIC_SERVO_CMD, command)

    def nudge(self, delta_degrees: float):
        """Move relative to the current angle -- used for live tracking."""
        self.move_to(self.current_angle + delta_degrees)

    def stop(self):
        print("[MQTT] -> ESP: STOP")
        self.client.publish(TOPIC_SERVO_CMD, "STOP")

    def home(self):
        print("[MQTT] -> ESP: HOME")
        self.client.publish(TOPIC_SERVO_CMD, "HOME")

    def publish_recognition(self, name, distance, similarity, angle, status="TARGET"):
        payload = {
            "name": name,
            "status": status,
            "distance": float(distance),
            "similarity": float(similarity),
            "angle": int(angle),
        }
        self.client.publish(TOPIC_RECOGNITION, json.dumps(payload))

    def publish_not_found(self):
        payload = {"name": None, "status": "NOT_FOUND"}
        self.client.publish(TOPIC_RECOGNITION, json.dumps(payload))

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


# ============================================================
# FACE ANALYSIS + CLASSIFICATION
# ============================================================

def analyze_frame(frame, detector, embedder, matcher) -> List[Tuple[object, MatchResult]]:
    """Returns a list of (face, MatchResult) for every face detected in the frame."""
    faces = detector.detect(frame, max_faces=5)
    results = []

    for face in faces:
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        embedding = embedder.embed(aligned)
        result = matcher.match(embedding)
        results.append((face, result))

    return results


def classify(result: MatchResult, target_name: Optional[str]) -> str:
    """TARGET / OTHER_KNOWN / STRANGER for a single match result."""
    if result.accepted and target_name is not None and result.name == target_name:
        return "TARGET"
    if result.accepted:
        return "OTHER_KNOWN"
    return "STRANGER"


def find_target(results, target_name: Optional[str]):
    """Returns (face, result) for the target if present among results, else None."""
    if target_name is None:
        return None
    for face, result in results:
        if classify(result, target_name) == "TARGET":
            return face, result
    return None


def face_offset_normalized(face, frame_w: int) -> float:
    """Horizontal offset of a face's center from the frame's center, -1 (left) to +1 (right)."""
    cx = (face.x1 + face.x2) / 2.0
    offset = (cx - frame_w / 2.0) / (frame_w / 2.0)
    return float(np.clip(offset, -1.0, 1.0))


# ============================================================
# DRAWING -- vivid status overlays
# ============================================================

STATUS_COLORS = {
    "TARGET": (0, 255, 0),
    "OTHER_KNOWN": (0, 255, 255),
    "STRANGER": (0, 0, 255),
}


def draw_results(frame, results, target_name: Optional[str]):
    vis = frame.copy()

    for face, result in results:
        status = classify(result, target_name)
        color = STATUS_COLORS[status]

        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        for x, y in face.kps.astype(int):
            cv2.circle(vis, (int(x), int(y)), 2, color, -1)

        if status == "TARGET":
            label = f"TARGET: {result.name}"
        elif status == "OTHER_KNOWN":
            label = f"KNOWN: {result.name}"
        else:
            label = "STRANGER"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(face.y1, th + 10)
        cv2.rectangle(vis, (face.x1, ty - th - 10), (face.x1 + tw + 8, ty), color, -1)
        text_color = (0, 0, 0) if status != "STRANGER" else (255, 255, 255)
        cv2.putText(vis, label, (face.x1 + 4, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    return vis


def draw_banner(vis, text: str, color, pulse: bool = False):
    h, w = vis.shape[:2]
    bar_h = 46

    overlay = vis.copy()
    alpha = 0.55 + 0.25 * abs(np.sin(time.time() * 3.0)) if pulse else 0.75
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(vis, text, (max(10, (w - tw) // 2), bar_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return vis


def compute_banner(results, target_name: Optional[str], target_present: bool):
    """Priority: locked target > stranger present > other known present > searching."""
    if target_present:
        return f"TARGET LOCKED: {target_name}", (0, 140, 0), False

    statuses = [classify(r, target_name) for _, r in results]

    if "STRANGER" in statuses:
        return "STRANGER DETECTED", (0, 0, 200), True

    if "OTHER_KNOWN" in statuses:
        names = ", ".join(sorted({r.name for f, r in results if classify(r, target_name) == "OTHER_KNOWN"}))
        return f"KNOWN (not target): {names}", (0, 150, 150), False

    label = f"SEARCHING for {target_name}..." if target_name else "SEARCHING..."
    return label, (0, 0, 180), True


def render_frame(frame, results, target_name, target_present, extra_text=None):
    vis = draw_results(frame, results, target_name)
    text, color, pulse = compute_banner(results, target_name, target_present)
    if extra_text:
        text = extra_text
    vis = draw_banner(vis, text, color, pulse=pulse)
    return vis


def process_announcements(results, target_name, servo, angle, state):
    """
    Console + MQTT logging for strangers / other known faces, only firing on
    a NEW sighting (transition into view) so a continuous sweep doesn't spam
    the same person every step while they stay in frame.
    """
    statuses_now = [classify(r, target_name) for _, r in results]

    stranger_now = "STRANGER" in statuses_now
    if stranger_now and not state["stranger_active"]:
        r = next(r for f, r in results if classify(r, target_name) == "STRANGER")
        print(f"[SWEEP] Stranger detected at {angle} degrees (dist={r.distance:.3f})")
        servo.publish_recognition(name="Stranger", distance=r.distance,
                                   similarity=r.similarity, angle=angle, status="STRANGER")
    state["stranger_active"] = stranger_now

    known_now = {r.name for f, r in results if classify(r, target_name) == "OTHER_KNOWN"}
    for name in known_now - state["known_active"]:
        r = next(r for f, r in results if r.name == name and classify(r, target_name) == "OTHER_KNOWN")
        print(f"[SWEEP] Known face (not target) at {angle} degrees: {name}")
        servo.publish_recognition(name=name, distance=r.distance,
                                   similarity=r.similarity, angle=angle, status="OTHER_KNOWN")
    state["known_active"] = known_now


# ============================================================
# CONTINUOUS SWEEP -- one leg (e.g. 90 -> 0, or 0 -> 180)
# ============================================================

def continuous_sweep_phase(cap, detector, embedder, matcher, servo, target_name,
                            start_angle, end_angle, announce_state) -> Tuple[str, bool]:
    """
    Continuously steps the servo from start_angle to end_angle, checking the
    camera at every step. Returns (outcome, quit_requested) where outcome is
    "found" (target spotted -- servo stopped at that angle) or
    "not_found_reached_end" (reached end_angle with no sighting).
    """
    direction = 1 if end_angle >= start_angle else -1
    angle = start_angle

    servo.move_to(angle)
    time.sleep(SERVO_SETTLE_TIME)

    while True:
        ok, frame = cap.read()

        if ok:
            results = analyze_frame(frame, detector, embedder, matcher)
            target = find_target(results, target_name)

            vis = render_frame(frame, results, target_name, target_present=target is not None,
                                extra_text=(f"SEARCHING for {target_name}...  ({servo.current_angle}\u00b0)"
                                            if target_name else None))
            cv2.imshow("Falcon Eye", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit", True

            if target is not None:
                face, result = target
                print()
                print("=" * 60)
                print("TARGET FOUND!")
                print("Name:", result.name)
                print("Distance:", f"{result.distance:.3f}")
                print("Angle:", servo.current_angle)
                print("=" * 60)
                servo.publish_recognition(name=result.name, distance=result.distance,
                                           similarity=result.similarity,
                                           angle=servo.current_angle, status="TARGET")
                return "found", False

            process_announcements(results, target_name, servo, servo.current_angle, announce_state)

        if angle == end_angle:
            break

        angle += direction * SCAN_STEP_ANGLE
        if (direction > 0 and angle > end_angle) or (direction < 0 and angle < end_angle):
            angle = end_angle

        servo.move_to(angle)
        time.sleep(SCAN_STEP_SETTLE_TIME)

    return "not_found_reached_end", False


# ============================================================
# LOCK-ON: WATCH + LIVE TRACKING
# ============================================================

def chase_after_loss(cap, detector, embedder, matcher, servo, target_name, direction_sign) -> str:
    """
    The target just left frame. Rather than immediately giving up, keep
    pushing the servo further in `direction_sign` -- the actual physical
    direction the servo was already moving during the last successful
    tracking nudge, not a recomputed sign -- in short steps, checking the
    camera at each one, for up to CHASE_MAX_DEGREES. This is a guess at
    where they walked to, based on which way the servo was already
    heading right before they disappeared.

    Returns "found" (re-acquired -- caller should resume watch_and_track),
    "not_found" (chase exhausted, genuinely lost), or "quit".
    """
    traveled = 0
    start_angle = servo.current_angle

    print(f"[CHASE] {target_name} left frame near {start_angle} degrees -- "
          f"chasing toward where they were heading")

    while traveled < CHASE_MAX_DEGREES:
        prev_angle = servo.current_angle
        next_angle = prev_angle + direction_sign * CHASE_STEP_ANGLE
        clamped = max(0, min(180, next_angle))

        servo.move_to(clamped)
        traveled += abs(clamped - prev_angle)
        time.sleep(CHASE_STEP_SETTLE_TIME)

        ok, frame = cap.read()
        if not ok:
            continue

        results = analyze_frame(frame, detector, embedder, matcher)
        target = find_target(results, target_name)

        vis = render_frame(frame, results, target_name, target_present=target is not None,
                            extra_text=f"CHASING {target_name}...  ({servo.current_angle}\u00b0)")
        cv2.imshow("Falcon Eye", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"

        if target is not None:
            face, result = target
            print(f"[CHASE] Re-acquired {target_name} at {servo.current_angle} degrees")
            return "found"

        # Hit the physical end of travel -- no point continuing this way.
        if clamped in (0, 180):
            break

    print(f"[CHASE] Didn't find {target_name} within {CHASE_MAX_DEGREES} degrees of last sighting")
    return "not_found"


def watch_and_track(cap, detector, embedder, matcher, servo, target_name) -> bool:
    """
    Actively track target_name while visible, nudging the servo to follow
    their left/right movement in real time. When they leave frame, first
    chases a short distance further in the direction they were last
    drifting (the edge of frame they exited from) to try to reacquire
    them. Only if that chase comes up empty do we give up, log "Lost",
    and let the caller resume the normal sweep from wherever the servo
    now sits. Returns quit_requested.
    """
    missed = 0
    last_nudge_direction = 0  # actual physical sign of the last servo move, for chasing

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(FRAME_CHECK_INTERVAL)
            continue

        frame_w = frame.shape[1]
        results = analyze_frame(frame, detector, embedder, matcher)
        target = find_target(results, target_name)

        vis = render_frame(frame, results, target_name, target_present=target is not None)
        cv2.imshow("Falcon Eye", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True

        nudged = False

        if target is not None:
            face, result = target
            missed = 0

            offset = face_offset_normalized(face, frame_w)
            if abs(offset) > TRACK_DEADZONE:
                step = float(np.clip(offset * TRACK_GAIN, -TRACK_MAX_STEP, TRACK_MAX_STEP))
                prev_angle = servo.current_angle
                servo.nudge(SERVO_DIRECTION_SIGN * step)
                nudged = True
                # Record the servo's ACTUAL resulting direction, not a
                # recomputed sign -- this is what chase_after_loss will
                # continue, so it can never disagree with reality (clamping
                # at 0/180, rounding, etc. all wash out automatically).
                if servo.current_angle != prev_angle:
                    last_nudge_direction = 1 if servo.current_angle > prev_angle else -1

        else:
            missed += 1
            if missed >= MISSED_CHECKS_BEFORE_LOST:
                if last_nudge_direction != 0:
                    outcome = chase_after_loss(cap, detector, embedder, matcher, servo,
                                                target_name, last_nudge_direction)
                    if outcome == "quit":
                        return True
                    if outcome == "found":
                        missed = 0
                        continue  # back into normal tracking above

                print(f"[WATCH] Lost {target_name} -- resuming sweep in the same direction "
                      f"from {servo.current_angle} degrees")
                return False

        # If we just moved the servo, give it extra time to physically get
        # there before the next frame is grabbed -- otherwise the next
        # frame can be captured mid-turn (blur / momentarily out of frame),
        # which reads as a false "missed" detection and can spuriously
        # trip MISSED_CHECKS_BEFORE_LOST even though you never moved.
        time.sleep(TRACK_SETTLE_TIME if nudged else FRAME_CHECK_INTERVAL)


# ============================================================
# FULL-RANGE SEARCH (both legs -> confirm absence if neither finds them)
# ============================================================

def run_absence_confirming_search(cap, detector, embedder, matcher, servo, target_name) -> bool:
    """
    Sweeps the whole 0-180 range in two legs (current -> 0, then 0 -> 180),
    locking onto and following the target whenever seen. After they
    disappear, resumes the SAME leg in the SAME direction from wherever the
    servo currently is, rather than restarting. Only once both legs are
    covered with no sighting is absence confirmed (published over MQTT).
    Returns quit_requested.
    """

    print()
    print("=" * 60)
    print(f"STARTING FULL-RANGE SEARCH for {target_name or '(no target selected)'}")
    print("=" * 60)

    announce_state = {"stranger_active": False, "known_active": set()}
    legs = [0, 180]

    for leg_end in legs:
        while True:
            outcome, quit_requested = continuous_sweep_phase(
                cap, detector, embedder, matcher, servo, target_name,
                servo.current_angle, leg_end, announce_state
            )

            if quit_requested:
                return True

            if outcome == "not_found_reached_end":
                print(f"[SWEEP] Reached {leg_end} degrees -- not there in this direction")
                break

            # outcome == "found" -> lock on and watch until they leave
            quit_requested = watch_and_track(cap, detector, embedder, matcher, servo, target_name)
            if quit_requested:
                return True

            print(f"[SWEEP] Resuming search toward {leg_end} degrees")
            # loop back: continuous_sweep_phase resumes from servo.current_angle -> leg_end

    servo.publish_not_found()
    print()
    print("=" * 60)
    print("ABSENCE CONFIRMED -- not found across the full range, restarting from home")
    print("=" * 60)

    return False


# ============================================================
# TARGET SELECTION
# ============================================================

def select_target(matcher: FaceDBMatcher) -> Optional[str]:
    if not matcher._names:
        print("[WARNING] Face database is empty -- nobody to follow.")
        return None

    if TARGET_NAME:
        if TARGET_NAME in matcher._names:
            return TARGET_NAME
        print(f"[WARN] TARGET_NAME '{TARGET_NAME}' not found in database.")

    print("\nKnown identities in database:")
    for i, name in enumerate(matcher._names, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("Who should Falcon Eye follow? (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matcher._names):
            return matcher._names[int(choice) - 1]
        if choice in matcher._names:
            return choice
        print("Invalid choice, try again.")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FALCON EYE FACE RECOGNITION (full-range sweep, direction-following)")
    print("=" * 60)

    detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    embedder = ArcFaceEmbedderONNX(model_path=ARC_FACE_MODEL, input_size=(112, 112))

    db = load_db_npz(DB_PATH)
    matcher = FaceDBMatcher(db=db, dist_thresh=DISTANCE_THRESHOLD)

    print("[DB] Identities:", len(matcher._names))
    if matcher._names:
        print("[DB] Names:", ", ".join(matcher._names))
    else:
        print("[WARNING] Face database is empty!")

    target_name = select_target(matcher)
    if target_name:
        print(f"[TARGET] Falcon Eye will follow: {target_name}")
    else:
        print("[TARGET] No target -- will only report strangers/known faces, never lock on.")

    servo = ServoController()
    servo.move_to(HOME_ANGLE)
    time.sleep(0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        servo.close()
        raise RuntimeError("Camera not available")

    print()
    print("Camera ready")
    print("MQTT ready")
    print()
    print("Full-range search starts automatically. Press 'q' in the video window to quit.")

    try:
        while True:
            quit_requested = run_absence_confirming_search(cap, detector, embedder, matcher, servo, target_name)

            if quit_requested:
                break

            # absence confirmed across the full range -- return home and
            # start the whole sweep again automatically
            servo.move_to(HOME_ANGLE)
            time.sleep(SERVO_SETTLE_TIME)

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