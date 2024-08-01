from scripts.fruit_classifier import FruitClassifier
from scripts.health_classifier import HealthClassifier
from scripts.score_calculator import ScoreCalculator
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def build():
    # ML models
    # Build the fruit classifier
    fruit_classifier = FruitClassifier()
    fruit_classifier.build_model()
    fruit_classifier.train()
    fruit_classifier.evaluate()
    fruit_classifier.save_model()

    # Build the health classifier
    health_classifier = HealthClassifier()
    for fruit in ['apples', 'banana', 'oranges']:
        # Each fruit type has its own model
        health_classifier.build_model()
        health_classifier.train(fruit)
        health_classifier.evaluate(fruit)
        health_classifier.save_model(fruit)

    # NN models
    # Build the score calculator
    for fruit in ['apples', 'banana', 'oranges']:
        # Each fruit type has its own model
        score_calculator = ScoreCalculator(fruit)
        score_calculator.build_model()
        score_calculator.train_n_evaluate()
        score_calculator.save_model(f"../models/{fruit}_score.pth")


if __name__ == '__main__':
    build()