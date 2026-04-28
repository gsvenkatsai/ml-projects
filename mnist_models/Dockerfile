FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python + pip
RUN apt update && \
    apt install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# Copy rest of files
COPY . .

EXPOSE 5000

CMD ["python3", "predict.py"]