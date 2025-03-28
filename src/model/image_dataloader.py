import torch
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

IMAGE_SIZE = (128, 128)

class VA_Dataset(Dataset):
    def __init__(self, data_path, mapping_csv=None, transform=None):
        """
        Args:
            data_path (str): Path to directory containing emotion folders
            mapping_csv (str): Optional path to emotion mappings CSV
            transform (callable): Optional torchvision transforms
        """
        self.data_path = data_path
        self.transform = transform
        self.samples = []

        # Set default CSV path if not provided
        if mapping_csv is None:
            project_root = Path(__file__).parent.parent.parent
            mapping_csv = project_root / "data" / "emotion_mappings.csv"
        else:
            mapping_csv = Path(mapping_csv)

        # Verify CSV exists
        if not mapping_csv.exists():
            raise FileNotFoundError(
                f"Emotion mappings file not found at: {mapping_csv}\n"
                "Required format:\n"
                "emotion,valence,arousal\n"
                "anger,0.2,0.9\n"
                "happiness,0.9,0.7\n"
                "..."
            )

        # Load mappings
        self.mappings = pd.read_csv(mapping_csv, index_col='emotion')

        # Verify required columns
        if not {'valence', 'arousal'}.issubset(self.mappings.columns):
            raise ValueError(
                "CSV must contain 'valence' and 'arousal' columns\n"
                f"Found columns: {self.mappings.columns.tolist()}"
            )

        # Build dataset samples
        for emotion in os.listdir(data_path):
            emotion_path = os.path.join(data_path, emotion)
            if os.path.isdir(emotion_path) and emotion in self.mappings.index:
                label = self.mappings.loc[emotion, ['valence', 'arousal']].values
                label = torch.tensor(label, dtype=torch.float32)

                for img_file in os.listdir(emotion_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_path, img_file)
                        self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No valid images found in {data_path}\n"
                f"Supported emotions: {self.mappings.index.tolist()}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        return img, label


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Test configuration
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    try:
        # Initialize dataset
        dataset = VA_Dataset(
            data_path="../../data/emotion_images",  # Update this path
            transform=test_transform
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        images, labels = next(iter(dataloader))
        print(f"Batch shape - Images: {images.shape}, Labels: {labels.shape}")
        
        # Visualize
        plt.figure(figsize=(10, 5))
        for i in range(min(4, len(images))):
            img = images[i].numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            
            plt.subplot(1, 4, i+1)
            plt.imshow(img)
            plt.title(f"V: {labels[i][0]:.2f}\nA: {labels[i][1]:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify emotion_mappings.csv exists in ../../data/")
        print("2. Check your emotion_images folder structure")
        print("3. Ensure images are .jpg/.png format")