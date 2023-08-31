import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras.utils import to_categorical
#
# Loading Fashion MNIST training and test dataset
#
fashion_mnist_train = pd.read_csv('data/fashion-mnist_train.csv')
fashion_mnist_test = pd.read_csv('data/fashion-mnist_test.csv')
#
# Examining the shape of the data set
#
print(f'fashion_mnist_train.shape: {fashion_mnist_train.shape}')
print(f'fashion_mnist_test.shape: {fashion_mnist_test.shape}')

# Setting up the convolution neural network with convnet and maxpooling layer
#
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
# Model Summary
#
print(f'model.summary(): {model.summary()}')

# Adding the fully connected layers to CNN
#
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#
# Printing model summary
#
print(f'model.summary(): {model.summary()}')

# Configuring the network
#
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preparing the training data set for training
#
X = np.array(fashion_mnist_train.iloc[:, 1:])
y = to_categorical(np.array(fashion_mnist_train.iloc[:, 0]))
#
# Create training and validation data split
#
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Creating the test data set for testing
#
X_test = np.array(fashion_mnist_test.iloc[:, 1:])
y_test = to_categorical(np.array(fashion_mnist_test.iloc[:, 0]))
#
# Reshaping the dataset in (28, 28, 1) in order to feed into neural network
# Convnet takes the input tensors of shape (image_height, image_width, image_channels)
#
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
#
# Changing the dataset to float
#
X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
X_test = X_test.astype('float32')/255
#
# Examinging the shape of the dataset
#
print(f'X_train.shape: {X_train.shape}')
print(f'X_val.shape: {X_val.shape}')
print(f'X_test.shape: {X_test.shape}')

# Fit the CNN model
#
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=512)

import matplotlib.pyplot as plt
  
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
  
epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy vs Epochs
#
ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

# Evaluate the model accuracy and loss on the test dataset
#
test_loss, test_acc = model.evaluate(X_test, y_test)
#
# Print the loss and accuracy
#
print(f'test_loss: {test_loss}')
print(f'test_acc: {test_acc}')