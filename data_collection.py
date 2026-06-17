import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

CSV_FILE = 'dataset.csv'

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label'] + [f'pt{i}_{coord}' for i in range(21) for coord in ('x', 'y', 'z')]
        writer.writerow(headers)

letter = input("Enter the letter you want to collect (e.g. ALEF א, BEIT ב, GIMEL ג or SPACE): ").strip().upper()
num_samples = 100
samples_collected = 0

cap = cv2.VideoCapture(0)

print(f"Place your hand in front of the camera and make the sign for {letter}...")
print("Collection will start in 3 seconds...")
time.sleep(3)

print("Data collection started! Move your hand slightly (vary angles and distances)...")

while samples_collected < num_samples:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        
        row = [letter] + [val for p in lm for val in (p.x, p.y, p.z)]
        
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        samples_collected += 1
        
        mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Collecting {letter}: {samples_collected}/{num_samples}', 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Data Collection', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print(f"🎉 Done! Successfully collected {num_samples} samples for letter {letter} and saved to {CSV_FILE}.")