# Object Detection and Tracking

This application reads a webcam stream or video file, detects objects with a pretrained YOLO model, and assigns a stable tracking ID with a lightweight SORT-style tracker. Bounding boxes, labels, confidence scores, and IDs are drawn on every frame.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run with a webcam

```powershell
python main.py --source 0
```

## Run with a video file

```powershell
python main.py --source path\to\video.mp4
```

Press `q` while the video window is active to stop the program. On its first run, Ultralytics downloads the `yolov8n.pt` pretrained model. Use a larger compatible model through the `--model` option if accuracy is more important than speed.

## Tracking approach

The detector supplies bounding boxes for each frame. `SimpleSortTracker` matches them to active tracks using intersection-over-union (IoU). Matched detections keep their IDs, unmatched detections create new tracks, and stale tracks are removed after a short interval. This is intentionally small enough to make the tracking logic easy to understand and modify.
