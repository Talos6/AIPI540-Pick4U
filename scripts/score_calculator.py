import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from scripts.image_dataset import ImageDataset
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

class ScoreCalculator:
    def __init__(self, fruit):
        self.fruit = fruit
        self.model = None
        self.batch_size = 32
        self.num_workers = 4
        self.epoch = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    def build_model(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.to(self.device)

    def prepare_data(self):
        train_dataset = ImageDataset('train', self.fruit, transform=self.transform)
        test_dataset = ImageDataset('test', self.fruit, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, test_loader

    def train_n_evaluate(self):
        train_loader, test_loader = self.prepare_data()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        best_accuracy = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            val_accuracy = self.evaluate(test_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(f"../models/{self.fruit}_score.pth")

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def score(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image).squeeze()
            score = torch.sigmoid(output).item()
        return score

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(ROOT, path))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(ROOT, path)))
        self.model.to(self.device)
