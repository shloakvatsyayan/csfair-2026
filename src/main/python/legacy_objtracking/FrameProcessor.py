from legacy_objtracking.YoloUtils import YoloClassMapper, IouCalculator, BoxExtractor
from legacy_objtracking.ObjectTracker import ObjectTracker
from legacy_objtracking.BoxRenderer import BoxRenderer
from legacy_objtracking.MouseHandler import MouseClickHandler
from legacy_objtracking.DirectionDetector import DirectionDetector
from legacy_objtracking.TrackingHandlers import SimpleTrackingHandler


class FrameProcessor:

    def __init__(self, obj_class_name, iou_threshold_value=0.3):
        self._obj_class_name = obj_class_name
        self._iou_threshold_value = iou_threshold_value

        class_mapper = YoloClassMapper()
        iou_calculator = IouCalculator()
        self._box_extractor = BoxExtractor(class_mapper)
        self._object_tracker = ObjectTracker(iou_calculator, iou_threshold_value)
        self._box_renderer = BoxRenderer()

        direction_detector = DirectionDetector()
        tracking_handler = SimpleTrackingHandler(direction_detector)
        self._mouse_handler = MouseClickHandler(self._object_tracker, tracking_handler)
        self._tracking_handler = tracking_handler

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
            print_output=False
        )

