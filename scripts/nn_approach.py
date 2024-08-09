from PIL import ImageDraw, ImageFont
from scripts.fruit_detector import FruitDetector
from scripts.score_calculator import ScoreCalculator

class NNApproach:
    '''
    NN Approach: Fruit Detection -> Quality Score -> Top k Selection -> Response Drawing -> Output
    '''
    def process(self, image, k):
        fruit_detector = FruitDetector()
        fruit_detector.load_model()
        score_calculator = ScoreCalculator()
        score_calculator.build_model()
        identified_units = fruit_detector.detect_objects(image)

        scored_units = []
        for identified_unit in identified_units:
            x1, y1, x2, y2 = identified_unit['x1'], identified_unit['y1'], identified_unit['x2'], identified_unit['y2']
            unit_image = image.crop((x1, y1, x2, y2))
            fruit_label = identified_unit['label']
            fruit_class = fruit_label if fruit_label[-1] == 'a' else fruit_label + 's'
            score_calculator.load_model(f"../models/{fruit_class}_score.pth")
            score = score_calculator.score(unit_image)
            scored_units.append({
                'label': fruit_label,
                'score': score,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        scored_units = sorted(scored_units, key=lambda x: (-x['score'], (x['x2']-x['x1'])*(x['y2']-x['y1'])))

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for unit in scored_units[:k]:
            x1, y1, x2, y2 = unit['x1'], unit['y1'], unit['x2'], unit['y2']
            label = f"{unit['label']}: {unit['score']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            draw.text((x1, y1-10), label, fill='green', font=font)
        return image