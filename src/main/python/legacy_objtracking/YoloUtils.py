class YoloClassMapper:

    YOLO_CLASS_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(self):
        self._class_idx = {name: i for i, name in enumerate(self.YOLO_CLASS_NAMES)}

    def get_class_index(self, class_name):
        return self._class_idx.get(class_name, -1)

    def is_valid_class(self, class_name):
        return class_name in self._class_idx


class IouCalculator:

    def calculate(self, bounding_box_a, bounding_box_b):
        xA = max(bounding_box_a[0], bounding_box_b[0])
        yA = max(bounding_box_a[1], bounding_box_b[1])
        xB = min(bounding_box_a[2], bounding_box_b[2])
        yB = min(bounding_box_a[3], bounding_box_b[3])
        
        intersection_width = max(0, xB - xA)
        intersection_height = max(0, yB - yA)
        intersection_area = intersection_width * intersection_height
        
        area_a = (bounding_box_a[2] - bounding_box_a[0]) * (bounding_box_a[3] - bounding_box_a[1])
        area_b = (bounding_box_b[2] - bounding_box_b[0]) * (bounding_box_b[3] - bounding_box_b[1])
        
        return intersection_area / (area_a + area_b - intersection_area + 1e-6)


class BoxExtractor:

    def __init__(self, class_mapper):
        self._class_mapper = class_mapper

    def extract(self, model, frame, class_to_track):
        predictions = model.predict(source=frame, stream=True, verbose=False)
        current_frame_boxes = []
        target_class_idx = self._class_mapper.get_class_index(class_to_track)
        
        for prediction in predictions:
            for bounding_box in prediction.boxes:
                if int(bounding_box.cls[0]) == target_class_idx:
                    x1, y1, x2, y2 = map(int, bounding_box.xyxy[0])
                    current_frame_boxes.append((x1, y1, x2, y2))
        
        return current_frame_boxes
