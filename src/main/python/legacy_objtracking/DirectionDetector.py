class DirectionDetector:

    DIRECTION_NAMES = ["right", "left"]

    def __init__(self):
        pass

    def detect(self, tracked_box, frame_width):
        if not tracked_box:
            return None

        tracked_x1, _, tracked_x2, _ = tracked_box
        person_center_x = (tracked_x1 + tracked_x2) // 2
        frame_center_x = frame_width // 2

        direction_index = 0 if person_center_x > frame_center_x else 1
        return self.DIRECTION_NAMES[direction_index]
