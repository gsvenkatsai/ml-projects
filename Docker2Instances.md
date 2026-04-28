# Docker Build 1 vs Build 2 — MNIST Flask API (Ubuntu Base)

## Objective

Compare two ways to run the same ML inference service:

- **Build 1 (Manual)** using `docker build` + `docker run`
- **Build 2 (Compose)** using `docker compose`

Both serve a PyTorch CNN (MNIST) via a Flask API inside an Ubuntu-based container.

---

## Project Structure

```
~/ml-projects/mnist_models/
├── Dockerfile
├── docker-compose.yml   # only for Build 2
├── requirements.txt
├── predict.py
├── mnist_cnn.pth
```

---

## requirements.txt (CPU-only PyTorch)

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
flask
numpy
Pillow
```

**Why**: avoids downloading CUDA (GPU) libraries; keeps image smaller and builds faster.

---

## predict.py (Flask API)

```python
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

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

# Load model on CPU
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read()))

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return jsonify({
        "prediction": pred,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**Why**: converts a script into a long-running HTTP service.

---

## Dockerfile (Ubuntu Base)

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "predict.py"]
```

**Notes**:

- True Ubuntu base (manual Python install via `apt`)
- `COPY requirements.txt` first for layer caching
- `0.0.0.0` allows external access

---

# Build 1 — Manual (docker build + docker run)

## Build

```
docker build --no-cache -t mnist-api-ubuntu .
```

## Run

```
docker run -d -p 5000:5000 --name mnist-ubuntu mnist-api-ubuntu
```

## Test

```
curl -X POST -F "file=@test_digit.png" http://localhost:5000/predict
```

### Example Output

```
{"prediction": 6, "confidence": 0.7388697266578674}
```

```
curl -X POST -F "file=@test_digit.png" http://localhost:5000/predict
```

## Manage

```
docker stop mnist-ubuntu
docker start mnist-ubuntu
docker rm -f mnist-ubuntu
```

### Characteristics

- Imperative (manual commands)
- Good for quick tests and debugging
- Not easily reproducible

---

# Build 2 — Docker Compose (Declarative)

## docker-compose.yml

```yaml
services:
  mnist-api:
    build: .
    container_name: mnist-compose
    ports:
      - "5000:5000"
```

## Run

```
docker compose up -d --build
```

## Stop

```
docker compose down
```

## Logs

```
docker compose logs
```

## Test

```
curl -X POST -F "file=@test_digit.png" http://localhost:5000/predict
```

### Example Output

```
{"prediction": 6, "confidence": 0.7388697266578674}
```

```
curl -X POST -F "file=@test_digit.png" http://localhost:5000/predict
```

### Characteristics

- Declarative (defined in YAML)
- Reproducible and version-controlled
- Automatically handles build + run + networking

---
