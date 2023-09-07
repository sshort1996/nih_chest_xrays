from keras import layers
from keras import models
import streamlit as st


def compile_model():
    # Setting up the convolution neural network with convnet and maxpooling layer
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Model Summary
    st.write(f'model.summary(): {model.summary()}')

    # Adding the fully connected layers to CNN
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Printing model summary
    st.write(f'model.summary(): {model.summary()}')

    # Configuring the network
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model