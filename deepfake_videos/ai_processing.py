import os
import cv2
import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import shutil

# =======================
# 1️⃣ DOWNLOAD DATASET
# =======================
DATASET_URL = "https://drive.google.com/uc?id=1-k0B6_aN-P7NmFiMZ-8ajZQUnGEmRC6L"
DATASET_ZIP = "faceforensics.zip"
DATASET_DIR = "dataset"

if not os.path.exists(DATASET_DIR):
    print("Downloading dataset...")
    gdown.download(DATASET_URL, DATASET_ZIP, quiet=False)
    print("Extracting dataset...")
    os.system(f"unzip {DATASET_ZIP} -d {DATASET_DIR}")
    os.remove(DATASET_ZIP)  # Clean up ZIP file

# =======================
# 2️⃣ FRAME EXTRACTION
# =======================
def extract_frames(video_path, output_folder, frame_interval=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, frame_id = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
        count += 1

    cap.release()

# Extract frames from real and fake videos
extract_frames("dataset/real.mp4", "dataset/real_frames")
extract_frames("dataset/fake.mp4", "dataset/fake_frames")

# =======================
# 3️⃣ DATASET & DATALOADER
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root="dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =======================
# 4️⃣ DEFINE MODEL
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=True)

# Modify classifier for binary classification
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

model = model.to(device)

# =======================
# 5️⃣ TRAIN MODEL
# =======================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "deepfake_detector.pth")

# =======================
# 6️⃣ INFERENCE
# =======================
def predict_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(frame)
    
    return "FAKE" if prediction.item() > 0.5 else "REAL"

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count, fake_count, total_frames = 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            result = predict_frame(frame)
            total_frames += 1
            if result == "FAKE":
                fake_count += 1
            print(f"Frame {frame_count}: {result}")
        frame_count += 1

    cap.release()
    fake_percentage = (fake_count / total_frames) * 100
    print(f"Deepfake Probability: {fake_percentage:.2f}%")

# Example Usage
# detect_deepfake("test_video.mp4")

# =======================
# 7️⃣ CLEANUP DATASET
# =======================
def delete_dataset():
    shutil.rmtree(DATASET_DIR)
    print("Dataset deleted successfully.")

# Uncomment this line if you want to delete the dataset after training
# delete_dataset()
