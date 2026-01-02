import cv2


class MouseClickHandler:

    def __init__(self, object_tracker, tracking_handler):
        self._object_tracker = object_tracker
        self._tracking_handler = tracking_handler
        self._current_frame_boxes = []

    def set_current_frame_boxes(self, boxes):
        self._current_frame_boxes = boxes

    def handle_click(self, event, mouse_x, mouse_y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_box = self._find_clicked_box(mouse_x, mouse_y)
            
            if clicked_box:
                self._object_tracker.start_tracking(clicked_box)
                print("Selected new person for tracking:", clicked_box)
                self._tracking_handler.tracking_started("", clicked_box, mouse_x, mouse_y)
            else:
                self._object_tracker.stop_tracking()
                print("No person clicked. Tracking disabled.")
                self._tracking_handler.tracking_stopped("", None)

    def _find_clicked_box(self, mouse_x, mouse_y):
        for box_x1, box_y1, box_x2, box_y2 in self._current_frame_boxes:
            if box_x1 <= mouse_x <= box_x2 and box_y1 <= mouse_y <= box_y2:
                return (box_x1, box_y1, box_x2, box_y2)
        return None
