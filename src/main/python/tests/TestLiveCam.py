from ultralytics import YOLO
import cv2
import math
from legacy_objtracking.YoloUtils import YoloClassMapper


class VideoCaptureManager:

    def __init__(self, camera_index=0):
        self._camera_index = camera_index
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self._camera_index)

    def read_frame(self):
        if not self._cap:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()

    def wait_key(self, delay=1):
        return cv2.waitKey(delay)


class YoloModelLoader:

    def __init__(self, model_path):
        self._model_path = model_path

    def load(self):
        return YOLO(self._model_path)


class DetectionRenderer:

    def __init__(self):
        self._box_color = (255, 0, 255)
        self._box_thickness = 3
        self._text_color = (255, 0, 0)
        self._text_thickness = 2
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1

    def render_box(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._box_color, self._box_thickness)

    def render_label(self, frame, text, x, y):
        cv2.putText(
            frame, text, (x, y), self._font, self._font_scale,
            self._text_color, self._text_thickness
        )


class ObjectDetector:

    def __init__(self, class_mapper):
        self._class_mapper = class_mapper

    def process_detections(self, results):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_index = int(box.cls[0])
                class_name = self._class_mapper.YOLO_CLASS_NAMES[class_index]
                yield (x1, y1, x2, y2, confidence, class_name)


class LiveCamApplication:

    def __init__(self, camera_index=0, model_path="../../../../models/yolo11n.pt"):
        self._camera_index = camera_index
        self._model_path = model_path
        self._video_manager = VideoCaptureManager(camera_index)
        self._model_loader = YoloModelLoader(model_path)
        self._class_mapper = YoloClassMapper()
        self._detector = ObjectDetector(self._class_mapper)
        self._renderer = DetectionRenderer()
        self._model = None

    def initialize(self):
        self._model = self._model_loader.load()
        self._video_manager.start()

    def run(self):
        try:
            while True:
                success, img = self._video_manager.read_frame()
                if not success:
                    break

                results = self._model(img, stream=True)

                for x1, y1, x2, y2, confidence, class_name in self._detector.process_detections(results):
                    self._renderer.render_box(img, x1, y1, x2, y2)
                    self._renderer.render_label(img, class_name, x1, y1)
                    print("Confidence --->", confidence)
                    print("Class name -->", class_name)

                cv2.imshow('Webcam', img)
                if self._video_manager.wait_key(1) == ord('q'):
                    break
        finally:
            self._cleanup()

    def _cleanup(self):
        self._video_manager.release()
        cv2.destroyAllWindows()


def main():
    app = LiveCamApplication(camera_index=0)
    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
