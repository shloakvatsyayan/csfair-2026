import cv2


class BoxRenderer:

    def __init__(self):
        self._default_color = (255, 0, 255)
        self._default_thickness = 2
        self._tracked_color = (0, 255, 0)
        self._tracked_thickness = 3

    def render(self, frame, boxes, tracked_box=None):
        for box in boxes:
            x1, y1, x2, y2 = box
            
            if tracked_box and box == tracked_box:
                color = self._tracked_color
                thickness = self._tracked_thickness
            else:
                color = self._default_color
                thickness = self._default_thickness

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
