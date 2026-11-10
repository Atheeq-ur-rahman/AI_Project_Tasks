"""A small IoU-based multi-object tracker inspired by SORT."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    label: str
    missed_frames: int = 0


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate IoU for boxes stored as [left, top, right, bottom]."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class SimpleSortTracker:
    """Assign track IDs by greedily matching same-class boxes with IoU."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 20):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections: list[tuple[np.ndarray, str]]) -> list[Track]:
        """Update active tracks from boxes and return their current state."""
        for track in self.tracks:
            track.missed_frames += 1

        candidates = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, (box, label) in enumerate(detections):
                if track.label == label:
                    candidates.append(
                        (intersection_over_union(track.box, box), track_index, detection_index)
                    )

        used_tracks, used_detections = set(), set()
        for score, track_index, detection_index in sorted(candidates, reverse=True):
            if score < self.iou_threshold:
                break
            if track_index in used_tracks or detection_index in used_detections:
                continue
            track = self.tracks[track_index]
            track.box = detections[detection_index][0]
            track.missed_frames = 0
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for detection_index, (box, label) in enumerate(detections):
            if detection_index not in used_detections:
                self.tracks.append(Track(self.next_id, box, label))
                self.next_id += 1

        self.tracks = [
            track for track in self.tracks if track.missed_frames <= self.max_missed_frames
        ]
        return self.tracks
