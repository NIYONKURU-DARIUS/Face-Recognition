# Falcon Eye — Face Recognition + Servo-Tracking Turret

A CPU-only face recognition pipeline (Haar detection → 5-point MediaPipe landmarks →
ArcFace ONNX embedding → enrollment/recognition) that drives a servo turret over MQTT:
when the camera loses sight of a known face, the servo sweeps through a set of angles
until it finds one, then holds position and keeps watching.

```
Camera → Haar detection → 5-point landmarks → alignment (112×112)
       → ArcFace ONNX embedding → face database → Known / Stranger
       → MQTT → ESP32 or ESP8266 → servo
```

## Repository Layout

```
Week1--Falcon Eye/
├── face-recognition-5pt/          # Python recognition & tracking pipeline
│   ├── data/
│   │   ├── enroll/<name>/         # aligned enrollment crops per identity (gitignore)
│   │   ├── debug_aligned/         # alignment debug snapshots (gitignore)
│   │   └── db/                    # face_db.npz / face_db.json (gitignore)
│   ├── models/
│   │   ├── embedder_arcface.onnx  # ~166 MB — too large for GitHub, don't commit
│   │   └── face_landmarker.task   # MediaPipe landmark model
│   ├── src/
│   │   ├── camera.py              # webcam smoke test
│   │   ├── detect.py              # Haar detection test
│   │   ├── landmarks.py           # 5-point landmark test
│   │   ├── align.py               # alignment test (112×112 warp)
│   │   ├── embed.py               # ArcFace ONNX embedding test
│   │   ├── enroll.py              # multi-identity enrollment tool
│   │   ├── evaluate.py            # threshold tuning (FAR/FRR sweep)
│   │   ├── recognize.py           # Falcon Eye scanner: recognition + MQTT + servo logic
│   │   ├── haar_5pt.py            # Haar + FaceMesh 5-point detector, alignment math
│   │   ├── face_signals.py        # Part 2: EAR, blink count, eyes-closed, and smile score
│   │   ├── face_tracking.py       # Part 2: Identity lock, spatial tracking & signal feed
│   │   └── main.py                # ESP8266 MicroPython firmware (flash to the board, not run on desktop)
│   ├── init_project.py
│   └── book/                      # full write-up of the pipeline design
└── sketch_sep11a/
    └── sketch_sep11a.ino          # ESP32 Arduino firmware (alternative to src/main.py)
```

## How It Works

### Part 1: Face Recognition + Sweep Turret
**Recognition side (desktop, Python):** `src/recognize.py` runs the full pipeline in a
loop. It sweeps the servo through a fixed set of angles (`SCAN_ANGLES`), pausing at each
one to check for a known face. If it finds one, it stops scanning and holds that
position. If it sees an unrecognized face, it logs/publishes "Stranger" and keeps
scanning. It talks to the servo controller entirely over MQTT — it never touches the
board's GPIO directly.

**Servo side (microcontroller):** either `src/main.py` (ESP8266, MicroPython) or
`sketch_sep11a/sketch_sep11a.ino` (ESP32, Arduino) subscribes to `falcon/eye/servo/cmd`,
accepts `ANGLE:<0-180>`, `STOP`, and `HOME` commands, and drives a standard hobby servo
via PWM. Status is published back on `falcon/eye/servo/status`.

### Part 2: Face Tracking with Identity Lock & Facial Signals
`src/face_tracking.py` introduces persistent tracking and gesture analysis:
1. **Strict Identity Lock:** Locks only the specified enrolled identity (`--target <name>`). Discards strangers, distractors, and other enrolled identities.
2. **State Machine:**
   * `SEARCHING`: Scans candidate faces until the target identity is confirmed via ArcFace.
   * `LOCKED`: Tracks the active target across frames using IoU and normalized center displacement; reverifies identity every `verify_every` frames.
   * `LOST`: Holds state during temporary occlusions (`lost_timeout` frames) before returning to `SEARCHING`.
3. **Facial Signals (`src/face_signals.py`):**
   * **Eye Aspect Ratio (EAR):** Computes vertical-to-horizontal eyelid distances.
   * **Blink Counting:** Emits a discrete blink event when eyes close and reopen.
   * **Eyes-Closed State:** Triggers `EYES CLOSED` after sustained eye closure ($\ge 8$ frames).
   * **Smile Detection:** Computes normalized mouth width vs. face width with hysteresis (`smile_on` / `smile_off`).
4. **Normalized Tracking Error:**
   * Outputs resolution-independent position errors `(error_x, error_y)` from $[-1.0, 1.0]$.
   * Implements a dead zone (`dead_zone=0.07`) and reports directional signals (`LEFT`, `RIGHT`, `UP`, `DOWN`, `CENTER`) for downstream motor control.
5. **Part 3 Motor Feed:** Produces software-only control signals via GUI overlay or `--signal` JSON stream.

**MQTT topics (Part 1)**

| Topic | Direction | Purpose |
|---|---|---|
| `falcon/eye/servo/cmd` | Python → board | `ANGLE:N`, `STOP`, `HOME` |
| `falcon/eye/servo/status` | board → Python | current angle / state |
| `falcon/eye/recognition` | Python → broker | recognition events |
| `falcon/eye/status/<client_id>` | board → broker | online/offline (last will) |

## ⚠️ Before you push this to GitHub

1. **WiFi credentials are hardcoded in plaintext** in both `src/main.py` and
   `sketch_sep11a.ino` (`WIFI_SSID` / `WIFI_PASS`). If this repo is public, anyone can
   read your network password. Move these to a gitignored config file (or at minimum
   change the password and rotate it before pushing) — don't commit the real values.
2. **`data/enroll/` contains real people's faces** (yours and at least one other
   person's), and `data/debug_aligned/` has more face crops. These are personal
   biometric images. Gitignore both folders — don't publish them, even to fulfill the
   assignment's "show enrollment of multiple identities" requirement. Screenshots or a
   description of the enrollment flow are a safer way to demonstrate that part.
3. **`models/embedder_arcface.onnx` is ~166 MB** — over GitHub's 100 MB hard limit. It
   must be gitignored; document how to download it instead (see Setup below).

## Setup (Python side)

1. **Virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
   ```
2. **Dependencies**
   ```bash
   pip install opencv-python numpy onnxruntime scipy tqdm mediapipe paho-mqtt
   ```
3. **Download the ArcFace ONNX model**
   ```bash
   curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
   unzip -o buffalo_l.zip
   cp w600k_r50.onnx models/embedder_arcface.onnx
   rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
   ```
4. **MediaPipe landmark model**: `models/face_landmarker.task` — obtain from
   [MediaPipe's model zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
   if not already present.
5. **Set your own WiFi/MQTT config** in a local, gitignored copy of `src/main.py` or
   `sketch_sep11a.ino` before flashing the board.

## Usage

### Part 1: Enrollment & Turret Recognition
```bash
python -m src.camera        # validate webcam
python -m src.detect        # validate face detection
python -m src.landmarks     # validate 5-point landmarks
python -m src.align         # validate alignment
python -m src.embed         # validate ArcFace embedding

python -m src.enroll        # enroll one or more identities
python -m src.evaluate      # tune the recognition threshold (needs 2+ identities)

python -m src.recognize     # run the Falcon Eye scanner (needs the servo board online)
```

Flash `src/main.py` to an ESP8266 or `sketch_sep11a.ino` to an ESP32 before running
`recognize.py`, so there's a servo controller listening on MQTT.

---

### Part 2: Face Tracking with Identity Lock

Run from `face-recognition-5pt`:
```bash
python -m src.face_tracking --target "Darius"
```

#### Interactive Controls:
* **`q`**: Quit the application.
* **`c`**: **Auto-calibrate neutral baseline** (calibrates your resting eye openness and mouth shape on the fly).

#### Key CLI Flags:
| Argument | Default | Description |
|---|---|---|
| `--target` | *(Required)* | Enrolled identity to lock onto |
| `--camera` | `0` | Camera device index |
| `--threshold` | `0.34` | ArcFace cosine distance acceptance threshold |
| `--ear-threshold` | `0.23` | EAR threshold for eye closure |
| `--smile-on` | `0.39` | Smile ratio threshold to enter `SMILE` |
| `--smile-off` | `0.36` | Smile ratio threshold to exit `SMILE` (hysteresis) |
| `--blink-min-frames`| `1` | Minimum frames below EAR threshold for a blink event |
| `--signal` | `False` | Headless mode: prints 1 JSON signal line per frame (Part 3 feed) |
| `--max-frames` | `0` | Exit after $N$ frames in `--signal` mode ($0 = \infty$) |
| `--width` / `--height` | `0` | Requested resolution ($0 = \text{driver default}$) |

#### Headless Mode (Part 3 Motor Feed)
```bash
python -m src.face_tracking --target "Darius" --signal
```
Outputs structured JSON lines to standard output for consumption by motor controllers:
```json
{
  "part": 2,
  "frame": 142,
  "lock_state": "LOCKED",
  "target": "Darius",
  "identity_locked": true,
  "error_x": -0.1523,
  "error_y": 0.0841,
  "horizontal": "LEFT",
  "vertical": "DOWN",
  "ear": 0.3102,
  "blink": false,
  "eyes_closed": false,
  "smiling": true,
  "smile_score": 0.4012,
  "blink_total": 4
}
```

---

## Full Write-Up

See [`face-recognition-5pt/book/`](./face-recognition-5pt/book) for the detailed
walkthrough of the recognition pipeline's design and threshold-tuning methodology.

## References

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin
  Loss for Deep Face Recognition.* CVPR 2019.
- InsightFace Project — 2D & 3D Face Analysis.
- ONNX / ONNX Runtime documentation.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building
  Perception Pipelines.*
- Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection Using Facial Landmarks.*