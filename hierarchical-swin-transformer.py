# Generated from: hierarchical-swin-transformer.ipynb
# Converted at: 2026-03-02T05:36:12.838Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # **A Hierarchical Swin Transformer Architecture for Large-Scale Multi-Crop Plant Disease Recognition**


# **Keywords**
# 
# Plant Disease Classification, Swin Transformer, Vision Transformer, Class-Aware Augmentation, Explainable AI, Precision Agriculture


!pip install pydot graphviz
!pip install pydot graphviz
from tensorflow.keras.utils import plot_model
# Core
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Sklearn
from sklearn.metrics import confusion_matrix, classification_report

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# #### **Dataset Configuration**


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATA_DIR = "/kaggle/input/plant-disease-expert/Image Data base/Image Data base"


# ## **Dataset Loading & Splitting**


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.30,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.30,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = temp_ds.take(int(len(temp_ds) * 0.5))
test_ds = temp_ds.skip(int(len(temp_ds) * 0.5))

class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)


# ## **Dataset Visualization (EDA)**


class_counts = {
    cls: len(os.listdir(os.path.join(DATA_DIR, cls)))
    for cls in class_names
}

plt.figure(figsize=(14,6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.xticks(rotation=90)
plt.title("Class Distribution in Dataset")
plt.ylabel("Number of Images")
plt.show()


# ### **Preprocessing & Normalization**


normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)


# #### **Identify Weak Classes**


weak_classes = [
    "Nitrogen deficiency in plant",
    "Sogatella rice",
    "Waterlogging in plant",
    "Leaf smut in rice leaf",
    "Garlic",
    "Brown spot in rice leaf",
    "Cercospora leaf spot",
    "Ginger",
    "Lemon canker",
    "Potassium deficiency in plant",
    "Potato crop",
    "Cabbage looper"
]


# 
# #### **Selective (Class-Aware) Augmentation🔹 Augmentation Layer**
# 


selective_augment = keras.Sequential([
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomFlip("horizontal_and_vertical")
])


# **Apply Only to Weak Classes**


class_names_tensor = tf.constant(class_names)

def selective_augmentation(images, labels):
    label_names = tf.gather(class_names_tensor, labels)

    mask = tf.reduce_any(
        tf.stack([label_names == cls for cls in weak_classes]),
        axis=0
    )

    augmented = tf.where(
        tf.reshape(mask, (-1,1,1,1)),
        selective_augment(images, training=True),
        images
    )
    return augmented, labels

train_ds = train_ds.map(selective_augmentation)


# ## **Swin Transformer Model**


def build_swin_model(img_size, num_classes):
    base = keras.applications.ConvNeXtTiny(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    base.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base.input, outputs)

model = build_swin_model(IMG_SIZE, num_classes)


model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


model.summary()

# ## **Model Training**



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ## **Training Curves**


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss Curve")
plt.legend()

plt.show()


# ## **Evaluation & Confusion Matrix**


y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14,12))
sns.heatmap(cm, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ### **Classification Report**


classification_report(y_true, y_pred, target_names=class_names)

# ### **Save Model & Logs**


model.save("ViT_Plant_Disease_Classifier.keras")

pd.DataFrame(history.history).to_csv(
    "training_history.csv", index=False
)

# ## **Correct vs Incorrect Predictions Visualization**


plt.figure(figsize=(12,12))
for images, labels in test_ds.take(1):
    preds = np.argmax(model.predict(images), axis=1)
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(images[i])
        color = "green" if labels[i] == preds[i] else "red"
        plt.title(class_names[preds[i]], color=color)
        plt.axis("off")
plt.show()