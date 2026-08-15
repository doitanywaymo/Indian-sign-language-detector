"""
train_dynamic.py
Trains dynamic ISL model on your collected dataset.
Uses same feature extractor as original project.
Output: dynamic_sign_model.h5
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_PATH = "dataset/dynamic"

# Labels must match folder names exactly
LABELS = sorted(os.listdir(DATA_PATH))
print(f"Labels found: {LABELS}")

SEQUENCE_LENGTH = 30
FEATURE_DIM     = 129

# ==========================
# Load dataset
# ==========================
X, y = [], []

for label_index, label in enumerate(LABELS):
    label_path = os.path.join(DATA_PATH, label)
    for person in os.listdir(label_path):
        person_path = os.path.join(label_path, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            if not file.endswith(".npy"):
                continue
            seq = np.load(os.path.join(person_path, file))
            if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                X.append(seq)
                y.append(label_index)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\nLoaded {len(X)} sequences across {len(LABELS)} classes")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl}: {np.sum(y == i)} sequences")

# Shuffle
X, y = shuffle(X, y, random_state=42)

# One-hot encode
y_cat = to_categorical(y, num_classes=len(LABELS))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ==========================
# Model
# ==========================
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(LABELS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ==========================
# Train
# ==========================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
    ModelCheckpoint('dynamic_sign_model_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True, verbose=0)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ==========================
# Evaluate
# ==========================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ==========================
# Save
# ==========================
model.save("dynamic_sign_model.h5")
print("Saved: dynamic_sign_model.h5")
print(f"Label order: {LABELS}")
print("\nIMPORTANT: Copy the label order above into app.py DYNAMIC_LABELS list")
