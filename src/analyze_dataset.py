import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PIL import Image
import os

def analyze_dataset():
    # Load dataset
    data = pd.read_excel('data/cleaned_data.xlsx')
    
    # 1. Disease Distribution
    plt.figure(figsize=(12, 6))
    disease_counts = data[['D', 'G', 'C', 'A', 'H']].sum()
    disease_counts.plot(kind='bar')
    plt.title('Distribution of Eye Diseases in Dataset')
    plt.xlabel('Diseases')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/disease_distribution.png')
    plt.close()
    
    # 2. Co-occurrence Matrix
    plt.figure(figsize=(10, 8))
    co_occurrence = data[['D', 'G', 'C', 'A', 'H']].corr()
    sns.heatmap(co_occurrence, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Disease Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig('outputs/co_occurrence_matrix.png')
    plt.close()
    
    # 3. Disease Combinations
    plt.figure(figsize=(12, 6))
    combinations = data[['D', 'G', 'C', 'A', 'H']].sum(axis=1).value_counts().sort_index()
    combinations.plot(kind='bar')
    plt.title('Number of Diseases per Patient')
    plt.xlabel('Number of Diseases')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig('outputs/disease_combinations.png')
    plt.close()
    
    # 4. Find example images for each disease
    diseases = {
        'D': 'Diabetic Retinopathy',
        'G': 'Glaucoma',
        'C': 'Cataract',
        'A': 'AMD',
        'H': 'Hypertensive Retinopathy'
    }
    
    example_images = {}
    for code, disease in diseases.items():
        # Find a case with only this disease
        mask = (data[code] == 1) & (data[[c for c in ['D', 'G', 'C', 'A', 'H'] if c != code]] == 0).all(axis=1)
        if mask.any():
            example_idx = data[mask].index[0]
            example_images[disease] = {
                'image_path': f"data/ODIR-5K/{data.loc[example_idx, 'Left-Fundus']}",
                'disease_code': code
            }
    
    return example_images

if __name__ == "__main__":
    example_images = analyze_dataset()
    print("\nExample Images for Each Disease:")
    for disease, info in example_images.items():
        print(f"\n{disease}:")
        print(f"Image: {info['image_path']}")
        print(f"Disease Code: {info['disease_code']}") 