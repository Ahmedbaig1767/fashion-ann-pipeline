import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

data = np.load("data/raw/fashion_mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Split a validation set out of the training data
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/fashion_mnist_processed.npz",
    x_train=x_train, y_train=y_train,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)