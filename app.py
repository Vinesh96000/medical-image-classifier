import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_dir = 'data'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Classes: {dataset.classes}")
images, labels = next(iter(dataloader))
print(f"Batch image tensor shape: {images.shape}")
print("starting training!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

print("\nTraining finished!\n")\


def classify_image(img_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img_tensor = data_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    print(f"Prediction: {dataset.classes[pred]}")

example_med = os.path.join(data_dir, dataset.classes[0], os.listdir(os.path.join(data_dir, dataset.classes[0]))[0])
example_non_med = os.path.join(data_dir, dataset.classes[1], os.listdir(os.path.join(data_dir, dataset.classes[1]))[0])
torch.save(model.state_dict(), "model.pt")

print("\nTest medical image:")
classify_image(example_med)
print("\nTest non-medical image:")
classify_image(example_non_med)

