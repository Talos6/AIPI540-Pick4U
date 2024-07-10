import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))

class FruitClassifier:
    def __init__(self):
        self.img_size = (100, 100)
        self.model = SVC(probability=True)
        self.label_dict = {
            'apples': 0, 
            'banana': 1, 
            'oranges': 2,
        }

    def build_model(self):
        self.model = SVC(probability=True)

    def load_data(self, type):
        images = []
        labels = []
        for fruit, label in self.label_dict.items():
            fruit_dirs = [
                os.path.join(ROOT, f'../data/fruits/{type}/fresh{fruit}/'),
                os.path.join(ROOT, f'../data/fruits/{type}/rotten{fruit}/')
            ]
            for fruit_dir in fruit_dirs:
                for filename in os.listdir(fruit_dir):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        img_path = os.path.join(fruit_dir, filename)
                        image = cv2.imread(img_path)
                        image = cv2.resize(image, self.img_size)
                        images.append(image)
                        labels.append(label)
        return images, labels

    def extract_features(self, images):
        features = []
        for img in images:
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
            features.append(hist)
        return np.array(features)

    def train(self):
        train_images, train_labels = self.load_data('train')
        train_features = self.extract_features(train_images)
        self.model.fit(train_features, train_labels)

    def evaluate(self):
        test_images, test_labels = self.load_data('test')
        test_features = self.extract_features(test_images)
        predictions = self.model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy: {accuracy}")

    def predict(self, features):
        prediction = self.model.predict(features)
        return prediction[0]

    def save_model(self):
        joblib.dump(self.model, os.path.join(ROOT, '../models/fruit_classifier.pkl'))

    def load_model(self):
        self.model = joblib.load(os.path.join(ROOT, '../models/fruit_classifier.pkl'))

    def get_fruit_name(self, label):
        return [ key for key, value in self.label_dict.items() if value == label][0]
