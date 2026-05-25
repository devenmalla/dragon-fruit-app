# -*- coding: utf-8 -*-

"""
Custom CNN Model
Dragon Fruit Disease Detection

- Trains a custom CNN from scratch
- Saves results inside: results/custom_cnn/
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================
# PATH SETUP (FIXED)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir   = os.path.join(BASE_DIR, "dataset", "val")
test_dir  = os.path.join(BASE_DIR, "dataset", "test")

# Save inside model-specific folder
results_dir = os.path.join(BASE_DIR, "results", "custom_cnn")
model_dir   = os.path.join(BASE_DIR, "models")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# LOAD DATA
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds   = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds  = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)

# =========================
# NORMALIZATION
# =========================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# MODEL
# =========================

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =========================
# TRAIN
# =========================

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# =========================
# SAVE MODEL
# =========================

model.save(os.path.join(model_dir, "custom_cnn_model.h5"))

# =========================
# PLOTS
# =========================

plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Custom CNN Accuracy")
plt.savefig(os.path.join(results_dir, "cnn_accuracy.png"))
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title("Custom CNN Loss")
plt.savefig(os.path.join(results_dir, "cnn_loss.png"))
plt.show()

# =========================
# EVALUATION
# =========================

y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(results_dir, "cnn_metrics.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Custom CNN Confusion Matrix")
plt.savefig(os.path.join(results_dir, "cnn_confusion.png"))
plt.show()