from deepface import DeepFace


class AIDetector:

    @staticmethod
    def _normalize_result(result):

        if isinstance(result, list):
            return result[0] if result else None

        if isinstance(result, dict):
            return result

        return None

    def analyze(self, face_frame):

        try:

            result = DeepFace.analyze(
                face_frame,
                actions=["age", "emotion"],
                enforce_detection=False
            )

            normalized = self._normalize_result(result)
            if not normalized:
                return None

            age = normalized.get("age")
            emotion = normalized.get("dominant_emotion", "N/A")
            emotion_scores = normalized.get("emotion", {})

            confidence = None
            if isinstance(emotion_scores, dict) and emotion in emotion_scores:
                confidence = float(emotion_scores[emotion])

            if isinstance(age, (float, int)):
                age = int(round(age))

            return {
                "age": age,
                "dominant_emotion": emotion,
                "confidence": confidence
            }

        except Exception as e:

            print("Detection error:", e)

            return None