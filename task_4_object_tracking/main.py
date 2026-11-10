"""Run YOLO object detection and draw simple tracking IDs on a video stream."""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import SimpleSortTracker


def parse_source(value: str):
    """Treat numeric source values as camera indexes and other values as file paths."""
    return int(value) if value.isdigit() else value


def colour_for_id(track_id: int) -> tuple[int, int, int]:
    """Produce a repeatable, readable BGR colour for a tracking ID."""
    return (
        (37 * track_id) % 180 + 50,
        (17 * track_id) % 180 + 50,
        (83 * track_id) % 180 + 50,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and track objects in a video stream.")
    parser.add_argument("--source", default="0", help="Camera index or video-file path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model file or model name")
    parser.add_argument("--confidence", type=float, default=0.45)
    args = parser.parse_args()

    model = YOLO(args.model)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = SimpleSortTracker()
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break

            result = model(frame, conf=args.confidence, verbose=False)[0]
            detections: list[tuple[np.ndarray, str]] = []
            detection_confidences: dict[tuple[int, int, int, int], float] = {}
            for box in result.boxes:
                coordinates = box.xyxy[0].cpu().numpy().astype(int)
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append((coordinates, label))
                detection_confidences[tuple(coordinates)] = confidence

            for track in tracker.update(detections):
                if track.missed_frames:
                    continue
                left, top, right, bottom = track.box.astype(int)
                colour = colour_for_id(track.track_id)
                confidence = detection_confidences.get(tuple(track.box.astype(int)), 0.0)
                cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
                text = f"{track.label} #{track.track_id} {confidence:.0%}"
                cv2.putText(
                    frame, text, (left, max(25, top - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 2, cv2.LINE_AA,
                )

            cv2.imshow("Object Detection and Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
