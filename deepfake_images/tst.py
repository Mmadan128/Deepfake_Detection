import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Define the same model structure as used during training
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

# Load the trained model
model_path = "deepfake_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model and load weights
model = DeepfakeCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load the test dataset
ds = load_dataset("thenewsupercell/celeb-df-image-dataset", split="test")

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Lists to store true and predicted labels
y_true = []
y_pred = []

# Run inference on the test set
with torch.no_grad():
    for example in ds:
        image = example["image"].convert("RGB")
        label = example["label"]  # 0 = Real, 1 = Fake

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get model prediction
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Store results
        y_true.append(label)
        y_pred.append(predicted_class)

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# Plot Accuracy Bar Chart
plt.figure(figsize=(5, 4))
plt.bar(["Accuracy"], [accuracy * 100], color="green")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")
plt.show()
