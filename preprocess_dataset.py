import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import os
import shutil
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def get_model(model_name, num_classes=5):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def organize_dataset():
    base_dir = 'data/ODIR-5K'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    for dir_path in [train_dir, test_dir]:
        for eye in ['left', 'right']:
            os.makedirs(os.path.join(dir_path, eye), exist_ok=True)
    df = pd.read_csv('data/full_df.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df])
    valid_rows = []
    for _, row in df.iterrows():
        left_src = os.path.join('data/preprocessed_images', row['Left-Fundus'])
        right_src = os.path.join('data/preprocessed_images', row['Right-Fundus'])
        split = row['split']
        left_dst = os.path.join(base_dir, split, 'left', row['Left-Fundus'])
        right_dst = os.path.join(base_dir, split, 'right', row['Right-Fundus'])
        left_exists = os.path.exists(left_src)
        right_exists = os.path.exists(right_src)
        if left_exists:
            shutil.copy2(left_src, left_dst)
        if right_exists:
            shutil.copy2(right_src, right_dst)
        # Sadece var olan dosyalar için kayıt tut
        if left_exists:
            row_left = row.copy()
            row_left['filename'] = row['Left-Fundus']
            valid_rows.append(row_left)
        if right_exists:
            row_right = row.copy()
            row_right['filename'] = row['Right-Fundus']
            valid_rows.append(row_right)
    cleaned_df = pd.DataFrame(valid_rows)
    cleaned_df.to_excel('data/cleaned_data.xlsx', index=False)
    print("Dataset organization completed!")

if __name__ == "__main__":
    organize_dataset()