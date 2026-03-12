from collections import Counter, deque
import math


class CentroidTracker:
    def __init__(self, max_distance=90, max_disappeared=8, reid_ttl_frames=45, reid_distance=120):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.reid_ttl_frames = reid_ttl_frames
        self.reid_distance = reid_distance
        self._next_id = 1
        self._frame_index = 0
        self._tracks = {}
        self._lost_tracks = {}

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _distance(c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def _register(self, box):
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box": box,
            "centroid": self._centroid(box),
            "disappeared": 0,
            "last_seen": self._frame_index,
        }
        return tid

    def _reactivate_or_register(self, box):
        centroid = self._centroid(box)
        best_id = None
        best_dist = None

        for tid, info in self._lost_tracks.items():
            if info["expires_at"] < self._frame_index:
                continue

            dist = self._distance(centroid, info["centroid"])
            if dist <= self.reid_distance and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = tid

        if best_id is not None:
            del self._lost_tracks[best_id]
            self._tracks[best_id] = {
                "box": box,
                "centroid": centroid,
                "disappeared": 0,
                "last_seen": self._frame_index,
            }
            return best_id

        return self._register(box)

    def _prune_lost_memory(self):
        stale = [tid for tid, info in self._lost_tracks.items() if info["expires_at"] < self._frame_index]
        for tid in stale:
            del self._lost_tracks[tid]

    def update(self, boxes):
        self._frame_index += 1
        self._prune_lost_memory()
        boxes = [tuple(map(int, b)) for b in boxes]

        if not boxes:
            to_remove = []
            for tid, info in self._tracks.items():
                info["disappeared"] += 1
                if info["disappeared"] > self.max_disappeared:
                    self._lost_tracks[tid] = {
                        "centroid": info["centroid"],
                        "expires_at": self._frame_index + self.reid_ttl_frames,
                    }
                    to_remove.append(tid)

            for tid in to_remove:
                del self._tracks[tid]

            return []

        if not self._tracks:
            assigned = []
            for box in boxes:
                tid = self._reactivate_or_register(box)
                assigned.append((tid, box))
            return assigned

        remaining_track_ids = set(self._tracks.keys())
        assignments = []

        for box in boxes:
            cx, cy = self._centroid(box)
            best_id = None
            best_dist = None

            for tid in remaining_track_ids:
                tx, ty = self._tracks[tid]["centroid"]
                dist = math.hypot(cx - tx, cy - ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist is not None and best_dist <= self.max_distance:
                self._tracks[best_id]["box"] = box
                self._tracks[best_id]["centroid"] = (cx, cy)
                self._tracks[best_id]["disappeared"] = 0
                self._tracks[best_id]["last_seen"] = self._frame_index
                remaining_track_ids.remove(best_id)
                assignments.append((best_id, box))
            else:
                tid = self._reactivate_or_register(box)
                assignments.append((tid, box))

        for tid in list(remaining_track_ids):
            self._tracks[tid]["disappeared"] += 1
            if self._tracks[tid]["disappeared"] > self.max_disappeared:
                self._lost_tracks[tid] = {
                    "centroid": self._tracks[tid]["centroid"],
                    "expires_at": self._frame_index + self.reid_ttl_frames,
                }
                del self._tracks[tid]

        return assignments


class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_history = {}
        self.confidence_history = {}

    def update(self, track_id, prediction):
        if not prediction:
            return None

        emo_q = self.emotion_history.setdefault(track_id, deque(maxlen=self.window_size))
        conf_q = self.confidence_history.setdefault(track_id, deque(maxlen=self.window_size))

        emotion = prediction.get("dominant_emotion")
        confidence = prediction.get("confidence")

        if isinstance(emotion, str) and emotion:
            emo_q.append(emotion)

        if isinstance(confidence, (int, float)):
            conf_q.append(float(confidence))

        smoothed_emotion = "N/A"
        if emo_q:
            smoothed_emotion = Counter(emo_q).most_common(1)[0][0]

        smoothed_confidence = None
        if conf_q:
            smoothed_confidence = sum(conf_q) / len(conf_q)

        return {
            "dominant_emotion": smoothed_emotion,
            "confidence": smoothed_confidence,
        }

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for store in (self.emotion_history, self.confidence_history):
            stale = [k for k in store.keys() if k not in active]
            for key in stale:
                del store[key]
