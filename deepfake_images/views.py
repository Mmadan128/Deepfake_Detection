import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import os

# Define the same model as used during training
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

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(settings.BASE_DIR, "deepfake_detector.pth")
model = DeepfakeCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    """ Runs deepfake detection on an uploaded image. """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return "Fake" if predicted_class == 1 else "Real"

def upload_image(request):
    """ Handles image upload and prediction. """
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        
        with open(file_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        prediction = predict_image(file_path)
        return render(request, "deepfake_images/result.html", {"prediction": prediction, "image_url": file_path})

    return render(request, "deepfake_images/upload.html")
