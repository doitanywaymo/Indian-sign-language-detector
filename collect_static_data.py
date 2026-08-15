import cv2
import os
import numpy as np
import mediapipe as mp
import time
from feature_extractor import extract_static_features

# ==========================
# CONFIG
# ==========================

DATA_PATH = "dataset/static"
PERSON_ID = PERSON_ID = input("Enter Person ID : ")
SAMPLES_PER_CLASS = 200
COUNTDOWN_TIME = 3
SAVE_EVERY_N_FRAMES = 4   # Controls collection speed (IMPORTANT)

# Exclude dynamic letters
labels = [ch for ch in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if ch not in ['H', 'J']]

# ==========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create folders
for label in labels:
    os.makedirs(os.path.join(DATA_PATH, label, PERSON_ID), exist_ok=True)

cap = cv2.VideoCapture(0)

print("\nStarting Static Data Collection\n")

# ==========================
# MAIN LOOP
# ==========================

for label in labels:

    print(f"\nPreparing for {label}")

    # --------------------------
    # Smooth Countdown
    # --------------------------
    start_time = time.time()
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = max(0, COUNTDOWN_TIME - int(elapsed))

        cv2.putText(frame, f"Get Ready for {label}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.putText(frame, str(remaining),
                    (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 0, 255), 5)

        cv2.imshow("Collecting Static Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if elapsed >= COUNTDOWN_TIME:
            break

    print(f"Collecting {label}...")

    count = 0
    frame_counter = 0

    # --------------------------
    # Data Collection
    # --------------------------
    while count < SAMPLES_PER_CLASS:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame_counter += 1

        if results.multi_hand_landmarks and frame_counter % SAVE_EVERY_N_FRAMES == 0:
            features = extract_static_features(results)

            # Ensure consistent feature length
            if features is not None and features.shape[0] == 126:

                save_path = os.path.join(
                 DATA_PATH, label, PERSON_ID, f"{count}.npy"
                )

                np.save(save_path, features)
                count += 1

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame,
                    f"{label} {count}/{SAMPLES_PER_CLASS}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Collecting Static Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"{label} completed.")

cap.release()
cv2.destroyAllWindows()

print("\nStatic Data Collection Completed Successfully!")