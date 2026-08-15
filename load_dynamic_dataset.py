import os
import numpy as np

DATA_PATH = "dataset/dynamic"

labels = sorted(os.listdir(DATA_PATH))

X = []
y = []

for label_index, label in enumerate(labels):

    label_path = os.path.join(DATA_PATH, label)

    for person in os.listdir(label_path):

        person_path = os.path.join(label_path, person)

        for file in os.listdir(person_path):

            file_path = os.path.join(person_path, file)

            sequence = np.load(file_path)

            X.append(sequence)
            y.append(label_index)

X = np.array(X)
y = np.array(y)

print("Dynamic dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)