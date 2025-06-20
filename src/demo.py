import gradio as gr
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model

def load_model(model_name, model_path):
    """Load the trained model"""
    model = get_model(model_name)
    # Get absolute path to model file
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the input image"""
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply transforms
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

def predict(image, model):
    """Make prediction on the input image"""
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)
    
    # Get disease names and probabilities
    diseases = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertensive Retinopathy']
    
    # Create structured results dictionary
    results = {}
    for i, disease in enumerate(diseases):
        prob = float(probs[0, i])
        prediction = "Positive" if prob >= 0.5 else "Negative"
        confidence = f"{prob:.2%}" # Format as percentage string
        results[disease] = {
            "Prediction": prediction,
            "Confidence": confidence
        }
    
    # Return only the structured results dictionary
    return results

def launch_demo(model_name='efficientnet_b0', model_path='outputs/best_efficientnet_b0.pth'):
    """Launch the Gradio demo interface"""
    # Load model
    model = load_model(model_name, model_path)
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=lambda img: predict(img, model),
        inputs=gr.Image(type="pil", label="Upload Fundus Image"),
        outputs=gr.JSON(label="Prediction Results"),
        title="Eye Disease Detection System",
        description="Upload a fundus image to detect potential eye diseases.",
        examples=[
            ["examples/normal.jpg"],
            ["examples/diabetic.jpg"],
            ["examples/glaucoma.jpg"]
        ],
        theme=gr.themes.Default()
    )
    
    # Launch the interface
    iface.launch(share=True)

if __name__ == "__main__":
    launch_demo(model_name='efficientnet_b0', model_path='outputs/best_efficientnet_b0.pth')