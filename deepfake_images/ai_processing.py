import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# --------------------------------
# Load Dataset from Hugging Face
# --------------------------------
print("Loading dataset...")
ds = load_dataset("thenewsupercell/celeb-df-image-dataset")

# --------------------------------
# Custom Dataset Wrapper
# --------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if isinstance(image, list):
            image = image[0]  # Fix for list-wrapped images
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# --------------------------------
# Image Transformations
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------------
# Load Data into PyTorch DataLoader
# --------------------------------
train_dataset = DeepfakeDataset(ds["train"], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# --------------------------------
# CNN Model Definition
# --------------------------------
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Dummy forward pass to compute input size for FC layer
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 2)  # 2 classes: Real vs Fake

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            x = self.pool(self.relu(self.conv1(x)))
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

# --------------------------------
# Training Function
# --------------------------------
def train(model, loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with loss and accuracy
            pbar.set_postfix(loss=f"{epoch_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

        # Print accuracy after each epoch
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy:.2f}% | Loss: {epoch_loss:.4f}")

    print("Training Complete!")

# --------------------------------
# Run Training
# --------------------------------
if __name__ == "__main__":
    model = DeepfakeCNN()
    train(model, train_loader)
