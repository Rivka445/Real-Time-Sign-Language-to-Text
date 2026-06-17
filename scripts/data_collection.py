import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import csv
import os
import time
import urllib.request

CSV_FILE = 'data/dataset.csv'
MODEL_PATH = 'artifacts/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        MODEL_PATH
    )
    print("Model downloaded.")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        headers = ['label']
        for hand_idx in range(2):
            for i in range(21):
                for coord in ('x', 'y', 'z'):
                    headers.append(f'h{hand_idx}_pt{i}_{coord}')
        writer.writerow(headers)

letter = input("Enter the letter you want to collect (e.g. ALEF, BEIT, GIMEL or SPACE): ").strip().upper()
num_samples = 100
samples_collected = 0

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

cap = cv2.VideoCapture(0)

print(f"Place your hand(s) in front of the camera and make the sign for {letter}...")
print("Collection will start in 3 seconds...")
time.sleep(3)
print("Data collection started! Move your hand slightly (vary angles and distances)...")

while samples_collected < num_samples:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        hands_data = []
        
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                hands_data.extend([val for p in lm for val in (p.x, p.y, p.z)])

                for connection in HAND_CONNECTIONS:
                    a, b = connection
                    ax, ay = int(lm[a].x * frame.shape[1]), int(lm[a].y * frame.shape[0])
                    bx, by = int(lm[b].x * frame.shape[1]), int(lm[b].y * frame.shape[0])
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
                for p in lm:
                    cx, cy = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            else:
                hands_data.extend([0.0] * 63)

        row = [letter] + hands_data

        with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(row)

        samples_collected += 1

    cv2.putText(frame, f'Collecting {letter}: {samples_collected}/{num_samples}',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"🎉 Done! Successfully collected {num_samples} samples for letter {letter} and saved to {CSV_FILE}.")