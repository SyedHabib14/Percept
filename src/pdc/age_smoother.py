"""Basic IoU tracker and quality-gated age rolling averages for live streams."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from pdc.association import calculate_iou

@dataclass
class AgeTrack:
    track_id: int
    box: tuple[int, int, int, int]
    last_seen: float
    ages: deque[float] = field(default_factory=deque)

class AgeSmoother:
    """SMA is default: its fixed five-frame window resists one-frame age spikes.

    EMA weights recent frames more strongly and reacts faster (Duke forecasting notes:
    https://people.duke.edu/~rnau/411avg.htm); replacing the mean below with an EMA
    is therefore a one-line alternative when responsiveness is preferred.
    """
    def __init__(self, window: int = 5, iou_threshold: float = 0.2, max_age_seconds: float = 1.0):
        self.window, self.iou_threshold, self.max_age_seconds = window, iou_threshold, max_age_seconds
        self._tracks: dict[int, AgeTrack] = {}; self._next_id = 1
    def update(self, box, raw_age: Optional[float], face_available: bool, face_crop_valid: bool, now: float):
        self.expire(now)
        candidates = [(calculate_iou(box, t.box), t) for t in self._tracks.values()]
        _, track = max(candidates, default=(0.0, None), key=lambda item: item[0])
        if track is None or calculate_iou(box, track.box) < self.iou_threshold:
            track = AgeTrack(self._next_id, box, now, deque(maxlen=self.window)); self._tracks[self._next_id] = track; self._next_id += 1
        track.box, track.last_seen = box, now
        if raw_age is not None and face_available and face_crop_valid: track.ages.append(float(raw_age))
        return track.track_id, (sum(track.ages) / len(track.ages) if track.ages else raw_age)
    def expire(self, now: float) -> None:
        self._tracks = {i:t for i,t in self._tracks.items() if now - t.last_seen <= self.max_age_seconds}
    def active_track_ids(self) -> set[int]:
        """Track ids currently live (not yet expired). Lets other per-frame,
        per-track state (e.g. classification-stability history) get pruned in
        step with this tracker instead of growing unboundedly."""
        return set(self._tracks.keys())