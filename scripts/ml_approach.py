import cv2
import numpy as np
import os
from scripts.fruit_classifier import FruitClassifier
from scripts.health_classifier import HealthClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))

class MLApproach:
    def image_preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def image_segmentation(self, image):
        edged = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def extract_features(self, image, contours):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, contours, -1, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        color_hist = cv2.calcHist([masked_image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        return color_hist
    
    def draw_bounding_box(self, image, contour, label):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    def process(self, image_path, k):
        fruit_classifier = FruitClassifier()
        fruit_classifier.load_model()
        image = cv2.imread(os.path.join(ROOT, '../data/input/', image_path))
        preprocessed_image = self.image_preprocess(image)
        contours = self.image_segmentation(preprocessed_image)

        identified_units = []

        for contour in contours:
            features = self.extract_features(image, contour)
            fruit_label = fruit_classifier.predict([features])
            fruit_name = fruit_classifier.get_fruit_name(fruit_label)
            health_classifier = HealthClassifier()
            health_classifier.load_model(fruit_name)
            health_score = health_classifier.predict([features])
            identified_units.append((contour, fruit_name, health_score))

        identified_units = sorted(identified_units, key=lambda x: x[2], reverse=True)
        for unit in identified_units[:k]:
            self.draw_bounding_box(image, unit[0], f"{unit[1]}: {unit[2]:.2f}")

        cv2.imwrite(os.path.join(ROOT, '../data/output/', image_path), image)
            

            