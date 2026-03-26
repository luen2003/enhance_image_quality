FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    flask \
    opencv-python-headless \
    "numpy<2" \
    pillow \
    torch==2.1.0 \
    torchvision==0.16.0 \
    basicsr==1.4.2 \
    realesrgan==0.3.0

EXPOSE 5000

CMD ["python", "app.py"]