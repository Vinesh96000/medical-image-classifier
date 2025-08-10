import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import io

st.set_page_config(
    page_title="Medical vs Non-Medical Classifier",
    layout="centered",
    page_icon="🩺"
)

# Minimal, gentle tweak for header and subtle image/result box
st.markdown(
    """
    <style>
        .highlight-box {
            border-radius: 12px;
            border: 1.5px solid #e3e8ef;
            box-shadow: 0 2px 10px 0 #e8eaed1a;
            padding: 1.3em;
            margin-top: 1.2em;
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Medical vs Non-Medical Image Classifier")
st.write(
    "<span style='color:#5683b8;font-weight:500;'>Built by VINESH  | AI Generalist</span>",
    unsafe_allow_html=True,
)
st.info("How to use: 1. Upload a medical or non-medical image. 2. View prediction & confidence. 3. Download your report!")

DATA_DIR = 'data'
CLASSES = ['medical', 'non_medical']
MODEL_PATH = 'model.pt'
MODEL_VERSION = "v1.0 | Trained July 2025"
st.caption(f"Model version: {MODEL_VERSION}")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def train_and_save_model():
    dataset = ImageFolder(DATA_DIR, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cpu")
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def load_trained_model():
    device = torch.device("cpu")
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

if not os.path.isfile(MODEL_PATH):
    st.warning("No trained model found — training now…")
    model = train_and_save_model()
    st.success("Model trained and saved!")
else:
    model = load_trained_model()

uploaded_file = st.file_uploader("Upload a medical or non-medical image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.container():
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Your uploaded image", use_container_width=True)
        img_tensor = data_transforms(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).cpu().numpy()[0]
            confidence = float(prob[0][pred])
            prediction = CLASSES[pred]
        st.markdown(
            f"<h4>Prediction: <span style='color:#226362;'>{prediction.capitalize()}</span> "
            f"<span style='font-size:18px;color:#7b809a;'>({confidence:.2%} confident)</span></h4>",
            unsafe_allow_html=True)
        # Downloadable report
        report = f"Prediction: {prediction}\nConfidence: {confidence:.2%}\n"
        st.download_button("Download Result as TXT", io.BytesIO(report.encode()), file_name="prediction.txt")
        # Minimal explainability/usage tip
        st.info("Tip: Medical images are often grayscale and high contrast; non-medical images are more variable in color.")
        st.markdown("</div>", unsafe_allow_html=True)
        if confidence > 0.99:
         st.info(f"🤖 Fun fact: Super-high AI confidence means your image's features are a near-perfect match to my training data for the '{prediction}' class.")



with st.expander("Project Info & Contact"):
    st.markdown(f"""
**Medical vs Non-Medical AI Classifier**  
- Built by Vinesh   
- Powered by PyTorch + Streamlit  
- [Connect on LinkedIn](https://www.linkedin.com/in/vinesh-j-b7b95025b/)
- Email: <span style='color:#226362;'>vineshreddy46@gmail.com</span>
""", unsafe_allow_html=True)

st.markdown("---")
st.write("© 2025 VINESH J. All rights reserved.")
st.markdown(
    "<div style='text-align:center;color:#6cc5ff;font-size:15px;font-weight:600;padding-top:4px;'>🚀 Powered by VynAI</div>",
    unsafe_allow_html=True
)
