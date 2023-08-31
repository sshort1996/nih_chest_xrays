import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import streamlit as st


def pre_process():

    fashion_mnist_train = pd.read_csv('MNIST_data/fashion-mnist_train.csv')
    fashion_mnist_test = pd.read_csv('MNIST_data/fashion-mnist_test.csv')
    #
    # Examining the shape of the data set
    #
    st.write(f'fashion_mnist_train.shape: {fashion_mnist_train.shape}')
    st.write(f'fashion_mnist_test.shape: {fashion_mnist_test.shape}')

    # Preparing the training data set for training
    #
    X = np.array(fashion_mnist_train.iloc[:, 1:])
    Y = to_categorical(np.array(fashion_mnist_train.iloc[:, 0]))
    #
    # Create training and validation data split
    #
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    #
    # Creating the test data set for testing
    #
    X_test = np.array(fashion_mnist_test.iloc[:, 1:])
    Y_test = to_categorical(np.array(fashion_mnist_test.iloc[:, 0]))
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
    st.write(f'X_train.shape: {X_train.shape}')
    st.write(f'X_val.shape: {X_val.shape}')
    st.write(f'X_test.shape: {X_test.shape}')

    return X_test, X_train, X_val, Y_test, Y_train, Y_val