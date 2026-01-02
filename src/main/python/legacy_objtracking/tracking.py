import threading
from legacy_objtracking.YoloUtils import YoloClassMapper, IouCalculator, BoxExtractor
from legacy_objtracking.ObjectTracker import ObjectTracker
from legacy_objtracking.BoxRenderer import BoxRenderer
from legacy_objtracking.MouseHandler import MouseClickHandler
from legacy_objtracking.TrackingHandlers import RobotTrackingHandler, NullTrackingHandler


class ClassNameReader:

    def __init__(self, frame_processor):
        self._frame_processor = frame_processor
        self._class_mapper = YoloClassMapper()

    def read_loop(self):
        while True:
            class_name = input("Class Name:")
            class_name = class_name.strip().lower()
            if self._class_mapper.is_valid_class(class_name):
                self._frame_processor.set_class_name(class_name)
                print(f"Class name set to: {class_name}")
            else:
                print(f"Invalid class name. Defaulting to: {self._frame_processor.get_class_name()}")
                print("Allowed names:")
                sorted_names = sorted(self._class_mapper.YOLO_CLASS_NAMES)
                for name in sorted_names:
                    print(f"-->{name}:")


class FrameProcessor:

    def __init__(self, obj_class_name, socket_client, iou_threshold_value=0.3):
        self._obj_class_name = obj_class_name
        self._iou_threshold_value = iou_threshold_value

        class_mapper = YoloClassMapper()
        iou_calculator = IouCalculator()
        self._box_extractor = BoxExtractor(class_mapper)
        self._object_tracker = ObjectTracker(iou_calculator, iou_threshold_value)
        self._box_renderer = BoxRenderer()

        try:
            from objtracking.pid import PIDController, FaceTrackerPIDController
            x_pid_controller = PIDController(Kp=0.1, Ki=0.00, Kd=0.05)
            y_pid_controller = PIDController(Kp=0.01, Ki=0.00, Kd=0.05)
            face_tracker_pid_controller = FaceTrackerPIDController(
                x_pid_controller, y_pid_controller, min_x_error=5, min_y_error=5
            )
            tracking_handler = RobotTrackingHandler(socket_client, face_tracker_pid_controller)
        except ImportError:
            tracking_handler = NullTrackingHandler()

        self._mouse_handler = MouseClickHandler(self._object_tracker, tracking_handler)
        self._tracking_handler = tracking_handler

        class_reader = ClassNameReader(self)
        threading.Thread(target=class_reader.read_loop, daemon=True).start()

    def set_class_name(self, class_name):
        self._obj_class_name = class_name

    def get_class_name(self):
        return self._obj_class_name

    def handle_mouse_click(self, event, mouse_x, mouse_y, flags, param):
        self._mouse_handler.handle_click(event, mouse_x, mouse_y, flags, param)

    def process_frame(self, frame, model):
        current_frame_boxes = self._box_extractor.extract(model, frame, self._obj_class_name)
        self._mouse_handler.set_current_frame_boxes(current_frame_boxes)

        tracking_success = self._object_tracker.update(current_frame_boxes)
        if not tracking_success and self._object_tracker.is_tracking():
            print("Lost track (no matching box over IOU threshold).")

        self._box_renderer.render(frame, current_frame_boxes, self._object_tracker.get_tracked_box())

        self._tracking_handler.turn_detection(
            self._object_tracker.is_tracking(),
            self._object_tracker.get_tracked_box(),
            frame,
            print_output=True
        )

