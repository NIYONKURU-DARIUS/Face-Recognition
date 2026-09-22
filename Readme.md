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
├── face-recognition-5pt/          # Python recognition pipeline
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
│   │   └── main.py                # ESP8266 MicroPython firmware (flash to the board, not run on desktop)
│   ├── init_project.py
│   └── book/                      # full write-up of the pipeline design
└── sketch_sep11a/
    └── sketch_sep11a.ino          # ESP32 Arduino firmware (alternative to src/main.py)
```

## How It Works

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

**MQTT topics**

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