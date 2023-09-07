import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import streamlit as st
from keras import layers
from keras import models
import matplotlib.pyplot as plt


#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join


# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, 
                training_images_filepath,
                training_labels_filepath,
                test_images_filepath, 
                test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

          # Convert data to NumPy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        return (x_train, y_train),(x_test, y_test)        
    

def compile_model():
    # Setting up the convolution neural network with convnet and maxpooling layer
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Model Summary
    print(f'model.summary(): {model.summary()}')

    # Adding the fully connected layers to CNN
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Printing model summary
    print(f'model.summary(): {model.summary()}')

    # Configuring the network
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model


def train_model(X_train,
                Y_train, 
                model):
    
    history = model.fit(X_train, Y_train,
                        epochs=15,
                        batch_size=512)
    
    return history


def analyse_history(model, history, X_test, Y_test):  

    history_dict = history.history
    loss_values = history_dict['loss']
    accuracy = history_dict['accuracy']
    
    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot the model accuracy vs Epochs
    ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
    # ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax[0].set_title('Training & Validation Accuracy', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Accuracy', fontsize=16)
    ax[0].legend()
    
    # Plot the loss vs Epochs
    ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
    # ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
    ax[1].set_title('Training & Validation Loss', fontsize=16)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)
    ax[1].legend()
    
    plt.show()

    # Evaluate the model accuracy and loss on the test dataset
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    
    # Print the loss and accuracy
    st.write(f'test_loss: {test_loss}')
    st.write(f'test_acc: {test_acc}')

    return test_loss, test_acc


if __name__ == "__main__":
    
    # Set file paths based on added MNIST Datasets
    input_path = '../MNIST_data/classic'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    print('initialised class')
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    print('loaded data')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print('mapped y data to categorical')
    # print(f'size of x_train: {len(x_train)}')

    model = compile_model()
    print('compiled model')

    history = train_model(x_train,
                    y_train, 
                    model)
    print('train model')

    test_loss, test_acc = analyse_history(model, history, x_test, y_test)
    print('analyse training')