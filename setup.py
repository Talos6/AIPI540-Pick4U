from scripts.fruit_classifier import FruitClassifier
from scripts.health_classifier import HealthClassifier

def build():
    fruit_classifier = FruitClassifier()
    print("Building model...")
    fruit_classifier.build_model()
    print("Training model...")
    fruit_classifier.train()
    print("Evaluating model...")
    fruit_classifier.evaluate()
    print("Saving model...")
    fruit_classifier.save_model()
    health_classifier = HealthClassifier()
    print("Building model for each fruit")
    for fruit in ['apples', 'banana', 'oranges']:
        health_classifier.build_model()
        print(f"Training model for {fruit}...")
        health_classifier.train(fruit)
        print(f"Evaluating model for {fruit}...")
        health_classifier.evaluate(fruit)
        print(f"Saving model for {fruit}...")
        health_classifier.save_model(fruit)

if __name__ == '__main__':
    build()