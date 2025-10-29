
# Create a comprehensive dashboard using Streamlit
dashboard_code = '''
"""
Elder Care Fall Detection Monitoring Dashboard
Real-time monitoring dashboard for fall detection system
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import ast

# Set page configuration
st.set_page_config(
    page_title="Elder Care Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #ffebee;
        padding: 15px;
        border-left: 5px solid #f44336;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 5px solid #4caf50;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
MODEL_PATH = r'E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\models\\fall_detection_model.pth'
TRAIN_CSV = r'E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\cleaned\\fall_dataset\\fall_train_labels_clean.csv'
TEST_CSV = r'E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\cleaned\\fall_dataset\\fall_test_labels_clean.csv'

# Class information
CLASS_NAMES = {0: 'Fall', 1: 'Standing', 2: 'Sitting'}
CLASS_COLORS = {0: '#f44336', 1: '#4caf50', 2: '#2196f3'}

# Model definition
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

# Image preprocessing
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model = FallDetectionModel(num_classes=3).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, device, checkpoint.get('val_acc', 0), None
        else:
            return None, None, 0, "Model file not found. Please train the model first."
    except Exception as e:
        return None, None, 0, f"Error loading model: {str(e)}"

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    try:
        train_df = pd.read_csv(TRAIN_CSV) if os.path.exists(TRAIN_CSV) else None
        test_df = pd.read_csv(TEST_CSV) if os.path.exists(TEST_CSV) else None
        
        if train_df is not None:
            train_df['class'] = train_df['label'].apply(lambda x: int(ast.literal_eval(x)[0]))
        if test_df is not None:
            test_df['class'] = test_df['label'].apply(lambda x: int(ast.literal_eval(x)[0]))
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None, None

def predict_image(image, model, device):
    """Make prediction on an image"""
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item()
            
            prob_dict = {CLASS_NAMES[i]: probabilities[0][i].item() for i in range(3)}
        
        return predicted_class, confidence_score, prob_dict
    except Exception as e:
        return None, None, None

def create_class_distribution_chart(df, title):
    """Create a pie chart for class distribution"""
    if df is not None:
        class_counts = df['class'].value_counts().sort_index()
        class_labels = [CLASS_NAMES[i] for i in class_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=class_labels,
            values=class_counts.values,
            marker=dict(colors=[CLASS_COLORS[i] for i in class_counts.index]),
            hole=0.3
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=True
        )
        
        return fig
    return None

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Elder Care Fall Detection Monitoring System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4320/4320371.png", width=100)
        st.title("Navigation")
        page = st.radio("Go to", ["Dashboard Overview", "Live Prediction", "Dataset Analysis", "Model Performance"])
        
        st.markdown("---")
        st.markdown("### System Status")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Load model and data
    model, device, val_acc, error = load_model()
    train_df, test_df = load_dataset_info()
    
    if error:
        st.error(error)
        st.info("Please run the training script first: `python train_fall_detection_model.py`")
    
    # Dashboard Overview Page
    if page == "Dashboard Overview":
        st.header("📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Model Accuracy",
                value=f"{val_acc:.2f}%" if model else "N/A",
                delta="High Performance" if val_acc > 85 else "Needs Improvement"
            )
        
        with col2:
            total_train = len(train_df) if train_df is not None else 0
            st.metric(
                label="Training Samples",
                value=total_train,
                delta=f"{total_train} images"
            )
        
        with col3:
            total_test = len(test_df) if test_df is not None else 0
            st.metric(
                label="Test Samples",
                value=total_test,
                delta=f"{total_test} images"
            )
        
        with col4:
            st.metric(
                label="Model Status",
                value="Ready" if model else "Not Loaded",
                delta="Active" if model else "Inactive"
            )
        
        st.markdown("---")
        
        # Class distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            if train_df is not None:
                fig = create_class_distribution_chart(train_df, "Training Set Distribution")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if test_df is not None:
                fig = create_class_distribution_chart(test_df, "Test Set Distribution")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        st.markdown("---")
        st.subheader("📈 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        if train_df is not None:
            with col1:
                st.markdown("**Training Set**")
                for cls_id, cls_name in CLASS_NAMES.items():
                    count = len(train_df[train_df['class'] == cls_id])
                    st.write(f"- {cls_name}: {count} samples")
        
        if test_df is not None:
            with col2:
                st.markdown("**Test Set**")
                for cls_id, cls_name in CLASS_NAMES.items():
                    count = len(test_df[test_df['class'] == cls_id])
                    st.write(f"- {cls_name}: {count} samples")
        
        with col3:
            st.markdown("**Detection Classes**")
            st.write("- 🔴 Fall: Emergency")
            st.write("- 🟢 Standing: Normal")
            st.write("- 🔵 Sitting: Normal")
    
    # Live Prediction Page
    elif page == "Live Prediction":
        st.header("🎯 Live Fall Detection")
        
        if not model:
            st.warning("Model not loaded. Please train the model first.")
            return
        
        st.markdown("Upload an image to detect fall, standing, or sitting positions.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                with st.spinner('Analyzing image...'):
                    pred_class, confidence, prob_dict = predict_image(image, model, device)
                
                if pred_class:
                    # Display prediction
                    if pred_class == "Fall":
                        st.markdown(f'<div class="alert-box">⚠️ <strong>ALERT: Fall Detected!</strong><br>Confidence: {confidence*100:.2f}%</div>', 
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box">✅ <strong>Status: {pred_class}</strong><br>Confidence: {confidence*100:.2f}%</div>', 
                                    unsafe_allow_html=True)
                    
                    # Probability chart
                    st.markdown("### Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Class': list(prob_dict.keys()),
                        'Probability': [v*100 for v in prob_dict.values()]
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        color='Class',
                        color_discrete_map={'Fall': '#f44336', 'Standing': '#4caf50', 'Sitting': '#2196f3'},
                        text='Probability'
                    )
                    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset Analysis Page
    elif page == "Dataset Analysis":
        st.header("📁 Dataset Analysis")
        
        if train_df is None and test_df is None:
            st.warning("No dataset information available.")
            return
        
        tab1, tab2 = st.tabs(["Training Set", "Test Set"])
        
        with tab1:
            if train_df is not None:
                st.subheader("Training Dataset")
                st.write(f"**Total Samples:** {len(train_df)}")
                
                # Display sample data
                st.markdown("### Sample Data")
                display_df = train_df[['image_name', 'class']].copy()
                display_df['class_name'] = display_df['class'].map(CLASS_NAMES)
                st.dataframe(display_df.head(10), use_container_width=True)
                
                # Class distribution
                st.markdown("### Class Distribution")
                class_dist = train_df['class'].value_counts().sort_index()
                for cls_id, count in class_dist.items():
                    percentage = (count / len(train_df)) * 100
                    st.write(f"**{CLASS_NAMES[cls_id]}:** {count} samples ({percentage:.2f}%)")
        
        with tab2:
            if test_df is not None:
                st.subheader("Test Dataset")
                st.write(f"**Total Samples:** {len(test_df)}")
                
                # Display sample data
                st.markdown("### Sample Data")
                display_df = test_df[['image_name', 'class']].copy()
                display_df['class_name'] = display_df['class'].map(CLASS_NAMES)
                st.dataframe(display_df.head(10), use_container_width=True)
                
                # Class distribution
                st.markdown("### Class Distribution")
                class_dist = test_df['class'].value_counts().sort_index()
                for cls_id, count in class_dist.items():
                    percentage = (count / len(test_df)) * 100
                    st.write(f"**{CLASS_NAMES[cls_id]}:** {count} samples ({percentage:.2f}%)")
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("🎯 Model Performance Metrics")
        
        if not model:
            st.warning("Model not loaded. Please train the model first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.write(f"**Architecture:** ResNet50")
            st.write(f"**Number of Classes:** 3")
            st.write(f"**Input Size:** 224x224")
            st.write(f"**Validation Accuracy:** {val_acc:.2f}%")
        
        with col2:
            st.subheader("Training Details")
            st.write(f"**Optimizer:** Adam")
            st.write(f"**Loss Function:** CrossEntropyLoss")
            st.write(f"**Data Augmentation:** Yes")
            st.write(f"**Transfer Learning:** Pre-trained on ImageNet")
        
        st.markdown("---")
        
        # Performance gauge
        st.subheader("Overall Performance")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=val_acc,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model Accuracy (%)"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
'''

# Save dashboard script
with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(dashboard_code)

print("Dashboard script created: dashboard.py")
print("\nTo run the dashboard:")
print("1. Install Streamlit: pip install streamlit plotly")
print("2. Run the dashboard: streamlit run dashboard.py")
print("\nThe dashboard will open in your web browser automatically.")
