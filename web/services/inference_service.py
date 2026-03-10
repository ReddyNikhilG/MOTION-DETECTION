from concurrent.futures import ThreadPoolExecutor

from web.services.web_detector import WebFaceAnalyzer


class InferenceService:
    def __init__(self, logs_dir, workers=2):
        self.analyzer = WebFaceAnalyzer(logs_dir)
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def analyze_image_data(self, image_data):
        frame = self.analyzer.decode_image(image_data)
        if frame is None:
            return None

        future = self.executor.submit(self.analyzer.analyze_frame, frame)
        return future.result(timeout=15)
