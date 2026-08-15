import numpy as np


# ==========================
# NORMALIZE HAND LANDMARKS
# ==========================

def normalize_hand(landmarks):

    landmarks = np.array(landmarks)

    # Translation invariance (center at wrist)
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale invariance
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))

    if max_dist > 0:
        landmarks = landmarks / max_dist

    return landmarks.flatten()


# ==========================
# STATIC FEATURES (126)
# ==========================

def extract_static_features(results):

    left_hand  = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            coords = []

            for lm in hand_landmarks.landmark:
                coords.append([lm.x, lm.y, lm.z])

            normalized = normalize_hand(coords)

            label = handedness.classification[0].label

            if label == "Left":
                left_hand = normalized
            else:
                right_hand = normalized

    return np.concatenate([left_hand, right_hand])


# ==========================
# DYNAMIC FEATURES (129)
# ==========================

def extract_dynamic_features(hand_results, face_results):

    static_features = extract_static_features(hand_results)

    nose = np.zeros(3)

    if face_results and face_results.multi_face_landmarks:

        nose_landmark = face_results.multi_face_landmarks[0].landmark[1]

        nose = np.array([
            nose_landmark.x,
            nose_landmark.y,
            nose_landmark.z
        ])

    return np.concatenate([static_features, nose])
