import csv
import cv2
import numpy as np
import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))

class NaiveApproach:
    '''
    Naive Approach: Template Matching -> Random Recommendation
    '''
    def __init__(self):
        self.template_size = (100, 100)
        self.max_samples = 20
        self.template_matching_threshold = 0.8
        self.templates = self.load_templates()

    def load_labels(self):
        '''
        Prepare class labels
        '''
        with open(os.path.join(ROOT, '../data/labeled/classname.txt'), 'r') as file:
            lables = {str(i): row.strip() for i, row in enumerate(file, start=0)}
        return lables

    def load_templates(self):
        '''
        Prepare matching templates
        '''
        label_dict = self.load_labels()
        templates = []
        counter = {classname: 0 for classname in label_dict.values()}

        with open(os.path.join(ROOT, '../data/labeled/train.csv')) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)

            for row in csvreader:
                image_path, label = row
                classname = label_dict[label]
                if counter[classname] <= self.max_samples:
                    template = cv2.imread(os.path.join(ROOT, '../data/labeled/', image_path), 0)
                    template_resize = cv2.resize(template, self.template_size)
                    templates.append((classname, template_resize))
                    counter[classname] += 1
                else:
                    continue

        return templates

    def detect_objects(self, image):
        '''
        Object dection by template matching
        '''
        detections = []
        
        for label, template in self.templates:
            # Template matching
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.template_matching_threshold)
            
            for pt in zip(*loc[::-1]):
                w, h = template.shape[::-1]
                bounding_box = (pt[0], pt[1], pt[0] + w, pt[1] + h)
                detections.append((bounding_box, label, 1.0))
        
        return detections

    def select_top_k(self, detections, k):
        '''
        Random selection of top k
        '''
        detections_sorted = sorted(detections, key=lambda x: x[2], reverse=True)
        
        if k > len(detections_sorted):
            k = len(detections_sorted)
        
        selected_detections = random.sample(detections_sorted, k)
        return selected_detections

    def draw_bounding_boxes(self, image, detections):
        '''
        Response by drawing bounding boxes
        '''
        for (bbox, label, score) in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image
    
    def process(self, image_path, k):
        '''
        Image load -> Greyscale -> Detection -> Selection -> Output
        '''
        print('image read')
        image = cv2.imread(os.path.join(ROOT, '../data/input/', image_path))
        print('image greyscale')
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('detection')
        detections = self.detect_objects(image_gray)
        print('selection')
        selection = self.select_top_k(detections, k)
        print('output')
        output_image = self.draw_bounding_boxes(image, selection)
        print('save')
        cv2.imwrite(os.path.join(ROOT, '../data/output/', image_path), output_image)
