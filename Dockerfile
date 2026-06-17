FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the MediaPipe hand landmark model at build time
RUN python -c "\
import urllib.request, os; \
os.makedirs('artifacts', exist_ok=True); \
urllib.request.urlretrieve(\
'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',\
'artifacts/hand_landmarker.task')"

CMD ["python", "live_prediction.py"]
