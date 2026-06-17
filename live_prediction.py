import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pickle
import numpy as np
import pandas as pd
import os

MODEL_FILE = 'model.pkl'
TASK_MODEL_PATH = 'hand_landmarker.task'

if not os.path.exists(MODEL_FILE):
    print(f"❌ Error: Model file '{MODEL_FILE}' not found! Please run the training script first.")
    exit()

print("🤖 Loading Random Forest model...")
with open(MODEL_FILE, 'rb') as f:
    classifier = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH),
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
print("📸 Camera opening... Perform sign language gestures in front of the screen! (Press 'q' to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    display_text = "No Hands Detected"

    if results.hand_landmarks:
        hands_data = []
        
        for idx in range(2):
            if idx < len(results.hand_landmarks):
                lm = results.hand_landmarks[idx]
                base_x = lm[0].x
                base_y = lm[0].y
                base_z = lm[0].z
                hands_data.extend([val for p in lm for val in (p.x - base_x, p.y - base_y, p.z - base_z)])
                
                for connection in HAND_CONNECTIONS:
                    a, b = connection
                    ax, ay = int(lm[a].x * frame.shape[1]), int(lm[a].y * frame.shape[0])
                    bx, by = int(lm[b].x * frame.shape[1]), int(lm[b].y * frame.shape[0])
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
                for p in lm:
                    cx, cy = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            else:
                hands_data.extend([0.0] * 63)
        
        input_data = pd.DataFrame([hands_data], columns=classifier.feature_names_in_)
        
        predicted_letter = classifier.predict(input_data)[0]
        probabilities = classifier.predict_proba(input_data)
        confidence = np.max(probabilities)
        
        if confidence > 0.5:
            display_text = f"Letter: {predicted_letter} ({confidence:.0%})"
        else:
            display_text = "Analyzing..."

    cv2.rectangle(frame, (10, 15), (480, 75), (0, 0, 0), -1)  # Black background banner
    cv2.putText(frame, display_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    cv2.imshow('Real-Time Sign Language Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("👋 Application closed.")