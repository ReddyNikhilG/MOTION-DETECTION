import unittest

from vision_utils import CentroidTracker, PredictionSmoother


class TrackerTests(unittest.TestCase):
    def test_tracker_keeps_id_for_nearby_face(self):
        tracker = CentroidTracker(max_distance=100)
        first = tracker.update([(100, 100, 60, 60)])
        second = tracker.update([(110, 105, 60, 60)])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0][0], second[0][0])

    def test_tracker_reassociates_id_after_short_disappearance(self):
        tracker = CentroidTracker(max_distance=80, max_disappeared=1, reid_ttl_frames=20, reid_distance=90)
        first = tracker.update([(200, 200, 50, 50)])
        original_id = first[0][0]

        tracker.update([])
        tracker.update([])
        reappeared = tracker.update([(210, 205, 50, 50)])

        self.assertEqual(len(reappeared), 1)
        self.assertEqual(reappeared[0][0], original_id)

    def test_tracker_does_not_reassociate_after_ttl_expires(self):
        tracker = CentroidTracker(max_distance=80, max_disappeared=1, reid_ttl_frames=2, reid_distance=90)
        first = tracker.update([(50, 50, 40, 40)])
        old_id = first[0][0]

        tracker.update([])
        tracker.update([])
        tracker.update([])
        tracker.update([])
        late_return = tracker.update([(55, 55, 40, 40)])

        self.assertEqual(len(late_return), 1)
        self.assertNotEqual(late_return[0][0], old_id)


class SmootherTests(unittest.TestCase):
    def test_smoother_averages_age_and_votes_emotion(self):
        smoother = PredictionSmoother(window_size=3)
        out1 = smoother.update(1, {"age": 20, "dominant_emotion": "happy", "confidence": 80})
        out2 = smoother.update(1, {"age": 22, "dominant_emotion": "happy", "confidence": 82})
        out3 = smoother.update(1, {"age": 24, "dominant_emotion": "neutral", "confidence": 84})

        self.assertIsNotNone(out1)
        self.assertEqual(out2["dominant_emotion"], "happy")
        self.assertEqual(out3["age"], 22)
        self.assertGreaterEqual(out3["confidence"], 80)


if __name__ == "__main__":
    unittest.main()
