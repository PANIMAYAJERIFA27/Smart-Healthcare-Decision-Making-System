# ================================
# IMPORT LIBRARIES
# ================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras as tf

import lstm   # your custom lstm module

# ================================
# LOAD DATA
# ================================
dataset = pd.read_csv('cardio_vascular.csv')
dataset = dataset.dropna(how="any")

# Convert multi-class to binary
dataset.loc[:, 'cardio'].replace([2, 3, 4], [1, 1, 1], inplace=True)

print(dataset.info())

# ================================
# DATA VISUALIZATION
# ================================

# Age vs Cholesterol
plt.figure(figsize=(4,4))
plt.bar(dataset['Age'], dataset['Chol'], color="orange")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Age vs Cholesterol")
plt.show()

# Age vs Chest Pain
plt.figure(figsize=(4,4))
plt.bar(dataset['Age'], dataset['Cp'], color="orange")
plt.xlabel("Age")
plt.ylabel("Chest Pain")
plt.title("Age vs Chest Pain")
plt.show()

# ================================
# SPLIT DATA
# ================================
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=121
)

# ================================
# MODEL BUILDING
# ================================
model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(lstm.lstm_layer(None, 64))
model.add(lstm.lstm_layer(None, 128))
model.add(lstm.lstm_layer(None, 254))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# ================================
# TRAIN MODEL
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    verbose=1
)

# ================================
# SAVE MODEL
# ================================
model.save("lstm_train.h5")

# ================================
# PREDICTION
# ================================
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# ================================
# METRICS
# ================================
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")
f1 = f1_score(y_test, y_pred, average="micro")
kappa = cohen_kappa_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", acc)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("Kappa    :", kappa)

# ================================
# CONFUSION MATRIX PLOT
# ================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ================================
# ACCURACY & LOSS PLOTS
# ================================
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.tight_layout()
plt.show()
