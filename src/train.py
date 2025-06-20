import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
from .utils import plot_loss, plot_confusion_matrix
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=15, device='cuda', model_name='model', patience=7, min_delta=1e-4):
    """
    Train the model with improved early stopping and learning rate scheduling
    """
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save best model
            os.makedirs('outputs', exist_ok=True)
            torch.save(model.state_dict(), f'outputs/best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'outputs/{model_name}_training_history.png')
    plt.close()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cuda', model_name='model'):
    """
    Evaluate the model on test data
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    precision = torch.mean((all_preds & all_labels).float() / (all_preds.float() + 1e-7))
    recall = torch.mean((all_preds & all_labels).float() / (all_labels.float() + 1e-7))
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # Calculate mAP
    ap_per_class = []
    for i in range(all_labels.shape[1]):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        ap_per_class.append(ap)
    mAP = np.mean(ap_per_class)
    
    print(f'\nTest Results for {model_name}:')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'mAP: {mAP:.4f}')

    # Plot confusion matrix for each class
    disease_labels = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertensive Retinopathy']
    for i, disease_label in enumerate(disease_labels):
        # Convert multi-label to binary for this class
        true_binary = all_labels[:, i]
        pred_binary = all_preds[:, i]
        
        # Calculate confusion matrix
        # Needs sklearn.metrics.confusion_matrix which is not imported in train.py
        # Assuming it will be imported or available globally for the plot function
        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(true_binary.numpy(), pred_binary.numpy())
        
        # Temporary placeholder for confusion matrix (assuming plot_confusion_matrix handles this)
        # In a real scenario, confusion_matrix from sklearn would be used here.
        # For demonstration, we'll just pass dummy data or rely on the plot function if it handles raw labels/preds.

        # Assuming plot_confusion_matrix can take true and predicted binary tensors directly
        # If not, confusion_matrix from sklearn is needed.
        try:
            # This requires sklearn.metrics.confusion_matrix, need to add import if not available
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_binary.numpy(), pred_binary.numpy())
            plot_confusion_matrix(cm, disease_label, f'outputs/{model_name}_confusion_matrix_{disease_label.replace(" ", "_")}.png')
        except ImportError:
            print("Scikit-learn not imported. Cannot plot confusion matrix.")
        except Exception as e:
            print(f"Error plotting confusion matrix for {disease_label}: {e}")

    
    return precision, recall, f1, mAP, all_preds, all_labels