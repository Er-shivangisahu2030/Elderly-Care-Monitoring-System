
# Create inference script for making predictions
inference_script = '''
"""
Fall Detection Model Inference Script
Elder Care Assistance and Monitoring System
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
MODEL_PATH = r'E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\models\\fall_detection_model.pth'

# Image preprocessing
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Model
class FallDetectionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(FallDetectionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Load model
def load_model(model_path):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = FallDetectionModel(num_classes=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_names = checkpoint.get('class_names', {0: 'Fall', 1: 'Standing', 2: 'Sitting'})
    return model, class_names

# Prediction function
def predict_image(image_path, model, class_names):
    """
    Predict the class of a single image
    
    Args:
        image_path: Path to the image file
        model: Trained model
        class_names: Dictionary mapping class indices to names
    
    Returns:
        predicted_class: Name of predicted class
        confidence: Confidence score
        probabilities: Dictionary of all class probabilities
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, None, None, f"Error loading image: {e}"
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get all probabilities
        prob_dict = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
    
    return predicted_class, confidence_score, prob_dict, None

# Batch prediction function
def predict_batch(image_paths, model, class_names):
    """
    Predict classes for multiple images
    
    Args:
        image_paths: List of image paths
        model: Trained model
        class_names: Dictionary mapping class indices to names
    
    Returns:
        results: List of prediction results
    """
    results = []
    for img_path in image_paths:
        pred_class, confidence, probs, error = predict_image(img_path, model, class_names)
        results.append({
            'image_path': img_path,
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probs,
            'error': error
        })
    return results

# Main execution
if __name__ == "__main__":
    print("Loading model...")
    model, class_names = load_model(MODEL_PATH)
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    
    # Example usage - single image prediction
    # Uncomment and modify the path below to test
    # test_image = r"path/to/your/test/image.jpg"
    # pred_class, confidence, probs, error = predict_image(test_image, model, class_names)
    # 
    # if error:
    #     print(f"Error: {error}")
    # else:
    #     print(f"\\nPrediction Results:")
    #     print(f"Predicted Class: {pred_class}")
    #     print(f"Confidence: {confidence*100:.2f}%")
    #     print(f"\\nAll Probabilities:")
    #     for cls, prob in probs.items():
    #         print(f"  {cls}: {prob*100:.2f}%")
'''

# Save inference script
with open('predict_fall_detection.py', 'w') as f:
    f.write(inference_script)

print("Inference script created: predict_fall_detection.py")
