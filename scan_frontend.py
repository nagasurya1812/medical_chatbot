import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Model Class
class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=6):  
        super(BrainTumorModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # ResNet50 for better accuracy
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the Trained Model
model_path = r"C:\Users\abish\OneDrive\Documents\Bot(medical)\medical_chatbot\dataset\Braintumor\brain_tumor_model.pth"

# Initialize Model
model = BrainTumorModel(num_classes=6).to(device)

try:
    # Load Model State Dict Properly
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # Allow small mismatches
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Tumor Classification Labels
tumor_classes = {
    0: ("No Tumor", "No abnormality detected."),
    1: ("Glioma Tumor - Stage 1", "Gliomas arise in glial cells of the brain."),
    2: ("Glioma Tumor - Stage 2", "Early-stage glioma detected, requires monitoring."),
    3: ("Meningioma Tumor - Stage 1", "Meningiomas are usually slow-growing."),
    4: ("Meningioma Tumor - Stage 2", "Moderate-stage meningioma detected."),
    5: ("Pituitary Tumor", "Pituitary tumors affect hormone production.")
}

# Streamlit UI
st.title("🧠 Brain Tumor Detection AI")
st.write("Upload a brain scan to check for tumors and get medical insights.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  

    # Preprocess Image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted = np.argmax(probabilities)
        confidence_score = probabilities[predicted] * 100  

    # Display Results
    tumor_type, medical_info = tumor_classes[predicted]
    st.markdown(f"### 🏥 **Prediction: {tumor_type}**")
    st.write(f"**Confidence Level:** {confidence_score:.2f}%")
    st.write(f"📌 **Medical Information:** {medical_info}")

    # Suggested Diagnosis & Treatment
    st.subheader("💡 Suggested Medical Advice")
    if predicted == 0:
        st.write("No tumor detected. Maintain a healthy lifestyle and consider regular check-ups.")
    elif predicted in [1, 2]:
        st.write("Glioma tumors may require MRI monitoring. Consult a neurologist for treatment options.")
    elif predicted in [3, 4]:
        st.write("Meningioma tumors can be treated with surgery or radiation therapy. Seek medical advice.")
    elif predicted == 5:
        st.write("Pituitary tumors can affect hormone levels. Consult an endocrinologist for further testing.")

    st.write("📌 *This AI model provides a preliminary diagnosis. Please consult a medical professional for an accurate assessment.*")

    # 📊 Visualizing Confidence Levels
    st.subheader("📊 Confidence Level Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [tumor_classes[i][0] for i in range(6)]
    
    bars = ax.barh(labels, probabilities * 100, color=["gray"] * 6)
    bars[predicted].set_color("red")  # Highlight predicted class

    ax.set_xlabel("Confidence (%)")
    ax.set_title("Tumor Classification Confidence Levels")
    ax.invert_yaxis()  # Invert to keep highest probability at top

    st.pyplot(fig)
