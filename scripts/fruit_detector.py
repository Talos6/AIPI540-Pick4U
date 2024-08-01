import os
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))

class FruitDetector:
    '''
    Fruit detector using YOLOv8
    '''
    def __init__(self):
        self.model = None

    def load_model(self):
        '''
        Load the YOLOv8 model
        '''
        self.model = YOLO(os.path.join(ROOT, '../models/yolov8n.pt'))
    
    def detect_objects(self, img):
        '''
        Detect objects for only fruits type
        '''
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
        
        
        
    