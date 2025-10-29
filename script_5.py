
# Create requirements.txt file
requirements = '''# Core ML/DL Libraries
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Dashboard
streamlit>=1.28.0

# Progress Bar
tqdm>=4.65.0

# Optional: For GPU support
# Install CUDA toolkit separately if using GPU
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("requirements.txt created!")
print("\nTo install all dependencies:")
print("pip install -r requirements.txt")
