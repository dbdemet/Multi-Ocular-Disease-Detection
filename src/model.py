import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def get_model(model_name='efficientnet_b0', pretrained=True):
    """
    Create and return the model architecture
    Args:
        model_name (str): Name of the model architecture
        pretrained (bool): Whether to use pretrained weights
    Returns:
        model (nn.Module): The model architecture
    """
    if model_name == 'efficientnet_b0':
        # Load pretrained EfficientNet-B0
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Modify the classifier for multi-label classification
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)  # 5 diseases
        )
        
    elif model_name == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
        
    elif model_name == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model