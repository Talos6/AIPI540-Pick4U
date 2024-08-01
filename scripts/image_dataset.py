import os
import torch
from torch.utils.data import Dataset
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))

class ImageDataset(Dataset):
    '''
    A custom torch dataset for data loading
    '''
    def __init__(self, type, fruit, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ['fresh', 'rotten']:
            dir_path = os.path.join(ROOT, f'../data/fruits/{type}/{label}{fruit}/')
            for image_name in os.listdir(dir_path):
                self.image_paths.append(os.path.join(dir_path, image_name))
                self.labels.append(0 if label == 'rotten' else 1)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)