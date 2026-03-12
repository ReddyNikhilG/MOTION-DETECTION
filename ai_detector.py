try:
    from deepface import DeepFace
except Exception:
    DeepFace = None


class AIDetector:

    def __init__(self):
        self.enabled = DeepFace is not None

    @staticmethod
    def _normalize_result(result):

        if isinstance(result, list):
            return result[0] if result else None

        if isinstance(result, dict):
            return result

        return None

    def analyze(self, face_frame):

        if not self.enabled:
            return {
                "dominant_emotion": "N/A",
                "confidence": None,
            }

        try:

            result = DeepFace.analyze(
                face_frame,
                actions=["emotion"],
                enforce_detection=False
            )

            normalized = self._normalize_result(result)
            if not normalized:
                return None

            emotion = normalized.get("dominant_emotion", "N/A")
            emotion_scores = normalized.get("emotion", {})

            confidence = None
            if isinstance(emotion_scores, dict) and emotion in emotion_scores:
                confidence = float(emotion_scores[emotion])

            return {
                "dominant_emotion": emotion,
                "confidence": confidence
            }

        except Exception as e:

            print("Detection error:", e)

            return None