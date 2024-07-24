import os
import cv2
from PIL import Image
from scripts.fruit_detector import FruitDetector
from scripts.score_calculator import ScoreCalculator

ROOT = os.path.dirname(os.path.abspath(__file__))

class NNApproach:
    
    def process(self, image_path, k):
        fruit_detector = FruitDetector()
        fruit_detector.load_model()
        identified_units = fruit_detector.detect_objects(os.path.join(ROOT, '../data/input/', image_path))

        image = Image.open(os.path.join(ROOT, '../data/input/', image_path))
        scored_units = []
        for identified_unit in identified_units:
            x1, y1, x2, y2 = identified_unit['x1'], identified_unit['y1'], identified_unit['x2'], identified_unit['y2']
            unit_image = image[y1:y2, x1:x2]
            fruit_label = identified_unit['label']
            score_calculator = ScoreCalculator(fruit_label)
            score_calculator.load_model()
            score = score_calculator.score(unit_image)
            scored_units.append({
                'label': fruit_label,
                'score': score,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        scored_units = sorted(scored_units, key=lambda x: x['score'], reverse=True)
        for unit in scored_units[:k]:
            x1, y1, x2, y2 = unit['x1'], unit['y1'], unit['x2'], unit['y2']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{unit['label']}: {unit['score']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imwrite(os.path.join(ROOT, '../data/output/', image_path), image)