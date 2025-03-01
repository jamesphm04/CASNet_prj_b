import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

class RealCratersDataset(Dataset):
    def __init__(self, root, train, test_size=0.1, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # Scan the root directory for images
        valid_extensions = (".jpg", ".jpeg", ".png")  # Adjust extensions as needed
        image_files = [f for f in os.listdir(root) if f.lower().endswith(valid_extensions)]

        # Generate fake labels for each image (for example, random binary labels)
        self.mapping = [(img, random.randint(0, 1)) for img in image_files]  # Random labels for binary classification

        # Split the data into train and test sets using the provided test_size
        if self.train:
            self.mapping, _ = train_test_split(self.mapping, test_size=test_size, random_state=42)
        else:
            _, self.mapping = train_test_split(self.mapping, test_size=test_size, random_state=42)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image_path, label = self.mapping[idx]
        full_path = os.path.join(self.root, image_path)

        # Load image and apply transformations (if any)
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
