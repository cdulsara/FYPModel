import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
import os

# Load your pre-processed dataset
X_train = np.load('preprocessed-data/X_train_large_rgb.npy')
X_test = np.load('preprocessed-data/X_test_large_rgb.npy')
y_train = np.load('preprocessed-data/y_train_large.npy')
y_test = np.load('preprocessed-data/y_test_large.npy')

# Convert string labels to integer labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode the integer labels
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Apply normalization for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Apply data augmentation to the training dataset
train_generator = train_datagen.flow(X_train, y_train_one_hot, batch_size=5)
test_generator = test_datagen.flow(X_test, y_test_one_hot, batch_size=5)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # Conv2D(128, (3, 3), activation='relu'),
    # MaxPooling2D((2, 2)),
    # Conv2D(256, (3, 3), activation='relu'),
    # MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# model.fit(X_train, y_train_one_hot, epochs=10, validation_split=0.2)

# Using the generator to fit the model
model.fit(train_generator, epochs=10, validation_data=test_generator)


# Predict probabilities for the test data
y_pred_proba = model.predict(X_test)

# Convert probabilities to label indices
y_pred = np.argmax(y_pred_proba, axis=1)

# For ROC-AUC score, you need the label probabilities of the positive class
# Ensure y_test_encoded is properly encoded to match the shape of y_pred_proba if needed

# Calculate Precision, Recall, F1-Score
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=2)
print('\nTest accuracy:', test_acc)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# ROC-AUC score calculation; this requires binary label indicators for multi-class format
# Convert y_test_encoded to binary format for ROC-AUC
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')

print(f'ROC-AUC Score: {roc_auc}')