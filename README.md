## Setup
`pip install flask opencv-python realesrgan basicsr torch torchvision numpy pillow`
`pip install -r requirements.txt`
## Run
`python app.py`
## Build Docker Image
`docker build -t ai-enhancer`
# Run Docker Container
`docker run -p 5000:5000 ai-enhancer`

