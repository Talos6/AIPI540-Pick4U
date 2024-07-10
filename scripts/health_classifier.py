import os
import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))

class HealthClassifier:
    def __init__(self):
        self.img_size = (100, 100)
        self.model = None
        self.label_dict = {
            'fresh': 1, 
            'rotten': 0
        }

    def build_model(self):
        self.model = GaussianNB()

    def load_data(self, fruit, type):
        images = []
        labels = []
        for condition, label in self.label_dict.items():
            data_dir = os.path.join(ROOT, f'../data/fruits/{type}/{condition+fruit}/')
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(data_dir, filename)
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

    def train(self, fruit):
        train_images, train_labels = self.load_data(fruit, 'train')
        train_features = self.extract_features(train_images)
        self.model.fit(train_features, train_labels)

    def evaluate(self, fruit):
        test_images, test_labels = self.load_data(fruit, 'test')
        test_features = self.extract_features(test_images)
        predictions = self.model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"{fruit} Health Accuracy: {accuracy}")

    def predict(self, features):
        return self.model.predict_proba(features)[0][1]

    def save_model(self, fruit):
        joblib.dump(self.model, os.path.join(ROOT, f'../models/{fruit}_health_classifier.pkl'))

    def load_model(self, fruit):
        self.model = joblib.load(os.path.join(ROOT, f'../models/{fruit}_health_classifier.pkl'))
