# -*- coding: utf-8 -*-

"""
MobileNetV2 - Transfer Learning
Saves results in: results/mobilenetv2/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =========================
# PATH SETUP (FIXED)
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir   = os.path.join(BASE_DIR, "dataset", "val")
test_dir  = os.path.join(BASE_DIR, "dataset", "test")

results_dir = os.path.join(BASE_DIR, "results", "mobilenetv2")
model_dir   = os.path.join(BASE_DIR, "models")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# DATA
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds   = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds  = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)

# =========================
# PREPROCESSING
# =========================

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds   = val_ds.map(lambda x, y: (preprocess(x), y))
test_ds  = test_ds.map(lambda x, y: (preprocess(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# MODEL
# =========================

base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
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

model.save(os.path.join(model_dir, "mobilenetv2_model.h5"))

# =========================
# PLOTS
# =========================

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("MobileNetV2 Accuracy")
plt.savefig(os.path.join(results_dir, "mobilenetv2_accuracy.png"))
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("MobileNetV2 Loss")
plt.savefig(os.path.join(results_dir, "mobilenetv2_loss.png"))
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

with open(os.path.join(results_dir, "mobilenetv2_metrics.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("MobileNetV2 Confusion Matrix")
plt.savefig(os.path.join(results_dir, "mobilenetv2_confusion.png"))
plt.show()