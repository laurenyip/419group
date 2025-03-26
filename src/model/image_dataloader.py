import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import os

IMAGE_SIZE = (128, 128)

class VA_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        
        # Load dataset
        for file in os.listdir(data_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                base_name = os.path.splitext(file)[0]
                valence_path = os.path.join(data_path, f"{base_name}_valence.txt")
                arousal_path = os.path.join(data_path, f"{base_name}_arousal.txt")
                
                if os.path.exists(valence_path) and os.path.exists(arousal_path):
                    self.samples.append((file, valence_path, arousal_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_file, valence_path, arousal_path = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.data_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Load labels
        with open(valence_path, 'r') as f:
            valence = float(f.read().strip())
        
        with open(arousal_path, 'r') as f:
            arousal = float(f.read().strip())
        
        # Normalize scores to 0-1 range
        valence = (valence - 1) / 9.0
        arousal = (arousal - 1) / 9.0
        
        # Convert to tensor
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor([valence, arousal], dtype=torch.float32)