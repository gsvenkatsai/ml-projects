import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFilter

# Same architecture as cnn_train.py
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)  # invert colors
])

def predict(image_path, label):
    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    print(f"[{label}] Predicted: {pred} | Confidence: {confidence:.2%}")

# Load original, create pixelated version on the fly
img = Image.open("test_digit.png")
img.save("normal_digit.png")

# Pixelate: shrink down then blow back up
small = img.resize((7, 7), Image.NEAREST)
pixelated = small.resize((28, 28), Image.NEAREST)
pixelated.save("pixelated_digit.png")

predict("normal_digit.png", "Normal")
predict("pixelated_digit.png", "Pixelated")