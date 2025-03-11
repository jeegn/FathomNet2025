import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MarineSpeciesDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Custom dataset for loading marine species images.

        Args:
            csv_file (str): Path to the annotations CSV file.
            image_dir (str): Path to the directory containing images (roi/).
            transform (callable, optional): Optional transform to apply to images.
        """
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        # Get unique species names and create a mapping to numeric labels
        self.classes = sorted(self.annotations["label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, os.path.basename(self.annotations.iloc[idx, 0]))
        label = self.annotations.iloc[idx, 1]

        # Convert label to index
        label_idx = self.class_to_idx[label]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label_idx