import os
import csv
import numpy as np
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

data = np.load("data/processed/fashion_mnist_processed.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

# Save training history to CSV
keys = list(history.history.keys())
with open("models/history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + keys)
    for i in range(len(history.history[keys[0]])):
        writer.writerow([i + 1] + [history.history[k][i] for k in keys])

print("Saved models/model.h5 and models/history.csv")