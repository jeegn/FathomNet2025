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


if __name__ == "__main__":
    # Set the correct path to your annotations CSV file
    csv_path = "/scratch/scholar/jdani/fathomNetData/annotations.csv"  # Update this path if necessary
    data_path = "/scratch/scholar/jdani/fathomNetData/roi"  # Update this path if necessary

    # Initialize the database
    database = MarineSpeciesDataset(csv_path, data_path)

    # Display histogram
    import matplotlib.pyplot as plt
    from collections import Counter

    # Save histogram to a file
    histogram_path = "marine_species_distribution.png"
    label_counts = Counter(database.annotations["label"])
    labels, counts = zip(*label_counts.items())
    
    print(len(database.classes))
    plt.figure(figsize=(12, 6))
    plt.barh(labels, counts)
    plt.xlabel("Number of Images")
    plt.ylabel("Species")
    plt.title("Distribution of Marine Species in Dataset")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    
    plt.savefig(histogram_path, bbox_inches="tight")

    print(f"Histogram saved at: {histogram_path}")