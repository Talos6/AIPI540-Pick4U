import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))

class FruitDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = YOLO(os.path.join(ROOT, '../models/yolov8n.pt'))
    
    def detect_objects(self, img):
        results = self.model.predict(source=img, classes=[46,47,49], conf=0.05)
        result_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                result_boxes.append({
                    'label': result.names[box.cls.item()],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
        return result_boxes
        
        
        
    