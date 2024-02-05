import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np


# Load your pre-processed dataset
X_train = np.load('preprocessed-data/X_train_large_rgb.npy')
X_test = np.load('preprocessed-data/X_test_large_rgb.npy')
y_train = np.load('preprocessed-data/y_train_large.npy')
y_test = np.load('preprocessed-data/y_test_large.npy')

# Define the CNN model architecture
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) # Helps prevent overfitting
    model.add(layers.Dense(len(np.unique(y_train)), activation='softmax')) # Ensure this matches the number of classes

    return model

# Assuming you've already loaded and preprocessed your data
input_shape = X_train[0].shape # Get the input shape from the preprocessed data

# Create the model
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Convert labels to numeric if they're not already
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print('\nTest accuracy:', test_acc)

# Optionally, save the model for later use
model.save('tea_leaf_classification_model.h5')
