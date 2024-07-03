import csv
import cv2
import numpy as np
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))

class NaiveApproach:
    def __init__(self):
        self.templates = self.load_templates()

    def load_templates(self):
        templates = {}

        with open(os.path.join(ROOT, '../data/labeled/train.csv')) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)

            for row in csvreader:
                image_path, label = row
                if label not in templates:
                    templates[label] = cv2.imread(os.path.join(ROOT, '../data/labeled/', image_path), 0)

        return templates

    def detect_objects(self, image):
        detections = []
        
        for label, template in self.templates.items():
            # Template matching
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                w, h = template.shape[::-1]
                bounding_box = (pt[0], pt[1], pt[0] + w, pt[1] + h)
                detections.append((bounding_box, label, 1.0))
        
        return detections

    def select_top_k(self, detections, k):
        detections_sorted = sorted(detections, key=lambda x: x[2], reverse=True)
        
        if k > len(detections_sorted):
            k = len(detections_sorted)
        
        selected_detections = random.sample(detections_sorted, k)
        return selected_detections

    def draw_bounding_boxes(self, image, detections):
        for (bbox, label, score) in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image
    
    def process(self, image_path, k):
        image = cv2.imread(os.path.join(ROOT, '../data/input/', image_path))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detect_objects(image_gray)
        selection = self.select_top_k(detections, k)
        output_image = self.draw_bounding_boxes(image, selection)
        cv2.imwrite(os.path.join(ROOT, '../data/output/', image_path), output_image)
