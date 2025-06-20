import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class ODIR5KDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train', eye='left'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            mode (string): 'train' or 'test'
            eye (string): 'left' or 'right'
        """
        self.data = pd.read_excel(csv_file)
        self.data = self.data[self.data['filename'].str.endswith(f'{eye}.jpg')].reset_index(drop=True)
        self.img_dir = os.path.join(img_dir, mode, eye)
        self.transform = transform
        self.mode = mode
        self.eye = eye
        
        # Define disease labels
        self.labels = ['D', 'G', 'C', 'A', 'H']  # Diabetic, Glaucoma, Cataract, AMD, Hypertensive
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Dosya bulunamazsa bir sonraki indeksi kullan
            return self.__getitem__((idx + 1) % len(self))
        labels = torch.tensor([
            self.data.iloc[idx][label] for label in self.labels
        ], dtype=torch.float32)
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, labels

# Define transforms
train_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.9, 1.1),
        rotate=(-45, 45),
        translate_percent=(-0.0625, 0.0625),
        p=0.5
    ),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, p=0.5),
    ], p=0.3),
    A.OneOf([
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])