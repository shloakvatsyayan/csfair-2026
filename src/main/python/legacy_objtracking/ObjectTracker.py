class ObjectTracker:

    def __init__(self, iou_calculator, iou_threshold=0.3):
        self._iou_calculator = iou_calculator
        self._iou_threshold = iou_threshold
        self._current_tracked_box = None
        self._is_tracking = False

    def start_tracking(self, bounding_box):
        self._current_tracked_box = bounding_box
        self._is_tracking = True

    def stop_tracking(self):
        self._current_tracked_box = None
        self._is_tracking = False

    def update(self, candidate_boxes):
        if not self._is_tracking or not self._current_tracked_box:
            return False

        best_iou_score = 0.0
        best_box_match = None

        for box in candidate_boxes:
            iou_value = self._iou_calculator.calculate(self._current_tracked_box, box)
            if iou_value > best_iou_score:
                best_iou_score = iou_value
                best_box_match = box

        if best_box_match and best_iou_score >= self._iou_threshold:
            self._current_tracked_box = best_box_match
            return True
        else:
            self.stop_tracking()
            return False

    def get_tracked_box(self):
        return self._current_tracked_box if self._is_tracking else None

    def is_tracking(self):
        return self._is_tracking
