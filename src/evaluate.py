import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots without opening a window
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

model = keras.models.load_model("models/model.h5")

data = np.load("data/processed/fashion_mnist_processed.npz")
x_test, y_test = data["x_test"], data["y_test"]

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Confusion matrix image
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

# Metrics file at project root
with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)

print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")