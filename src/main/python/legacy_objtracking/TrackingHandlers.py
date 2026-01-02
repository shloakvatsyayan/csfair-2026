class NullTrackingHandler:

    def tracking_started(self, obj_class_name, bounding_box, mouse_x, mouse_y):
        pass

    def tracking_stopped(self, obj_class_name, bounding_box):
        pass

    def box_updated(self, obj_class_name, bounding_box):
        pass

    def turn_detection(self, is_tracking, tracked_box, frame, print_output=False):
        pass


class SimpleTrackingHandler:

    def __init__(self, direction_detector):
        self._direction_detector = direction_detector

    def tracking_started(self, obj_class_name, bounding_box, mouse_x, mouse_y):
        pass

    def tracking_stopped(self, obj_class_name, bounding_box):
        pass

    def box_updated(self, obj_class_name, bounding_box):
        pass

    def turn_detection(self, is_tracking, tracked_box, frame, print_output=False):
        if is_tracking and tracked_box:
            direction = self._direction_detector.detect(tracked_box, frame.shape[1])
            if print_output and direction:
                print(direction)


class RobotTrackingHandler:

    def __init__(self, socket_client, pid_controller):
        self._client = socket_client
        self._pid_controller = pid_controller
        self._stopped_all = False

    def tracking_started(self, obj_class_name, bounding_box, mouse_x, mouse_y):
        pass

    def tracking_stopped(self, obj_class_name, bounding_box):
        pass

    def box_updated(self, obj_class_name, bounding_box):
        pass

    def turn_detection(self, is_tracking, tracked_box, frame, print_output=False):
        if not (is_tracking and tracked_box):
            if not self._stopped_all:
                self._client.send("es all")
                self._stopped_all = True
            return

        self._stopped_all = False
        tracked_x1, tracked_y1, tracked_x2, tracked_y2 = tracked_box
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        del_x, del_y = self._pid_controller.process(
            tracked_x1, tracked_y1, tracked_x2, tracked_y2, frame_w, frame_h
        )
        del_y = -del_y
        self._client.send(f"dc {del_x} {del_y}")
