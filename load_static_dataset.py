import os
import numpy as np

DATA_PATH = "dataset/static"

labels = sorted(os.listdir(DATA_PATH))

X = []
y = []

for label_index, label in enumerate(labels):

    label_path = os.path.join(DATA_PATH, label)

    for person in os.listdir(label_path):

        person_path = os.path.join(label_path, person)

        for file in os.listdir(person_path):

            file_path = os.path.join(person_path, file)

            data = np.load(file_path)

            X.append(data)
            y.append(label_index)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(labels))