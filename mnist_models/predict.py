from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# -------------------------
# Model (unchanged)
# -------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# -------------------------
# Load model
# -------------------------
device = torch.device("cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

# -------------------------
# API endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = Image.open(io.BytesIO(file.read()))
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return jsonify({
        "prediction": pred,
        "confidence": confidence
    })

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)