import cv2
from ultralytics import YOLO
from legacy_objtracking.FrameProcessor import FrameProcessor
from legacy_objtracking.DeviceDetector import DeviceDetector


class VideoCaptureManager:

    def __init__(self, camera_index=0):
        self._camera_index = camera_index
        self._camera_stream = None

    def start(self):
        print("Starting video capture...")
        self._camera_stream = cv2.VideoCapture(self._camera_index)

    def read_frame(self):
        if not self._camera_stream:
            return False, None
        return self._camera_stream.read()

    def release(self):
        if self._camera_stream:
            self._camera_stream.release()


class YoloModelLoader:

    def __init__(self, model_path, use_gpu=True):
        self._model_path = model_path
        self._use_gpu = use_gpu
        self._device_detector = DeviceDetector()

    def load(self):
        model = YOLO(self._model_path)
        if self._use_gpu:
            device = self._device_detector.get_device()
            model.to(device)
            device_info = self._device_detector.get_device_info()
            print(f"Model loaded on device: {device.upper()}")
            if device_info['is_mps']:
                print("Using MPS (Metal Performance Shaders) for GPU acceleration on macOS")
            elif device_info['is_cuda']:
                print("Using CUDA for GPU acceleration")
            else:
                print("Falling back to CPU")
        return model


class TrackingApplication:

    def __init__(self, class_to_track, model_path, camera_index=0, use_gpu=True):
        self._class_to_track = class_to_track
        self._model_loader = YoloModelLoader(model_path, use_gpu)
        self._video_manager = VideoCaptureManager(camera_index)
        self._frame_processor = None
        self._model = None

    def initialize(self):
        self._model = self._model_loader.load()
        self._video_manager.start()
        self._frame_processor = FrameProcessor(self._class_to_track)

        cv2.namedWindow("Webcam")
        cv2.setMouseCallback("Webcam", self._frame_processor.handle_mouse_click)
        print("Press 'q' to quit, started")

    def run(self):
        try:
            while True:
                ret, frame = self._video_manager.read_frame()
                if not ret:
                    break

                self._frame_processor.process_frame(frame=frame, model=self._model)

                cv2.imshow("Webcam", frame)
                key_pressed = cv2.waitKey(1)
                if key_pressed & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _cleanup(self):
        self._video_manager.release()
        cv2.destroyAllWindows()


def main():
    person = "person"
    class_to_track = person
    model_path = "models/yolo11n.pt"
    app = TrackingApplication(class_to_track, model_path, camera_index=0, use_gpu=True)
    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
