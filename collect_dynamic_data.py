import cv2
import os
import numpy as np
import mediapipe as mp
import time
from feature_extractor import extract_dynamic_features

# ==========================
# CONFIG
# ==========================

DATA_PATH = "dataset/dynamic"
PERSON_ID = input("Enter Person ID : ")
SEQUENCE_LENGTH = 30
SEQUENCES_PER_CLASS = 100
COUNTDOWN_TIME = 3

labels = [
    'H',
    'J',
    'hello',
    'bye',
    'namaste',
    'practice',
    'thank_you',
    'sorry'
]

# ==========================

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

for label in labels:
    os.makedirs(os.path.join(DATA_PATH, label, PERSON_ID), exist_ok=True)

cap = cv2.VideoCapture(0)

print("\nStarting Improved Dynamic Data Collection\n")

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
                    1, (0,255,255), 2)

        cv2.putText(frame, str(remaining),
                    (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0,0,255), 5)

        cv2.imshow("Collecting Dynamic Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if elapsed >= COUNTDOWN_TIME:
            break

    print(f"Collecting {label}")

    seq_num = 0

    while seq_num < SEQUENCES_PER_CLASS:

        sequence = []

        while len(sequence) < SEQUENCE_LENGTH:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb)
            face_results = face_mesh.process(rgb)

            if hand_results.multi_hand_landmarks:

                features = extract_dynamic_features(hand_results, face_results)

                # Ensure consistent feature length (IMPORTANT)
                if features.shape[0] == 129:
                    sequence.append(features)

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(frame,
                        f"{label} Seq:{seq_num+1}/{SEQUENCES_PER_CLASS}",
                        (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

            cv2.imshow("Collecting Dynamic Data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # 🔥 REDO CURRENT SEQUENCE
            if key == ord('r'):
                print("Redoing current sequence...")
                sequence = []

        # Save only if full sequence collected
        if len(sequence) == SEQUENCE_LENGTH:

            save_path = os.path.join(
                DATA_PATH, label, PERSON_ID, f"{seq_num}.npy"
            )

            np.save(save_path, np.array(sequence))
            seq_num += 1
            print(f"Saved sequence {seq_num}")

    print(f"{label} completed.")

cap.release()
cv2.destroyAllWindows()

print("\nDynamic Data Collection Completed Successfully!")