import torch
from torch.utils.data import DataLoader, Subset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import ODIR5KDataset, train_transforms, val_transforms
from src.model import get_model
from src.train import train_model, evaluate_model
from src.demo import launch_demo
from src.utils import generate_pdf_report
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load cleaned dataset to compute class weights
    data = pd.read_excel('data/cleaned_data.xlsx')
    labels = ['D', 'G', 'C', 'A', 'H']
    class_counts = {label: data[label].sum() for label in labels}
    total_patients = len(data)
    class_frequencies = [class_counts[label] / total_patients for label in labels]
    class_weights = [1.0 / freq for freq in class_frequencies]
    max_weight = max(class_weights)
    class_weights = torch.tensor([w / max_weight for w in class_weights]).to(device)
    print("Class Weights:", class_weights.tolist())
    
    # Load dataset
    train_dataset = ODIR5KDataset(
        csv_file='data/cleaned_data.xlsx',
        img_dir='data/ODIR-5K',
        mode='train',
        eye='right',
        transform=train_transforms
    )
    test_dataset = ODIR5KDataset(
        csv_file='data/cleaned_data.xlsx',
        img_dir='data/ODIR-5K',
        mode='test',
        eye='right',
        transform=val_transforms
    )
    
    # Split training dataset into train and validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    indices = np.random.permutation(len(train_dataset))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Define models to compare
    models_to_test = ['efficientnet_b0']
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTraining {model_name}...")
        model = get_model(model_name)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        # Train with improved parameters
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer,
            scheduler=scheduler,
            num_epochs=15,  # Increased max epochs
            device=device, 
            model_name=model_name,
            patience=7,  # Increased patience for early stopping
            min_delta=1e-4  # Minimum change in validation loss to be considered as improvement
        )
        
        # Evaluate
        precision, recall, f1, mAP, _, _ = evaluate_model(model, test_loader, device=device, model_name=model_name)
        results[model_name] = (precision, recall, f1, mAP)
        
        # Save model
        torch.save(model.state_dict(), f'outputs/model_{model_name}.pth')
    
    # Generate report
    generate_pdf_report(results, len(train_dataset) + len(test_dataset))
    
    # Launch demo with EfficientNet-B0
    print("\nLaunching demo with EfficientNet-B0")
    launch_demo(model_name='efficientnet_b0', model_path='outputs/model_efficientnet_b0.pth')

if __name__ == "__main__":
    main()