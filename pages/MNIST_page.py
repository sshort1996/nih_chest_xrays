import sys
sys.path.append("../")
from MNIST import preProcessor as pp
from MNIST import compile as cp
from MNIST import train as tr
from MNIST import test as ts
import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
import streamlit as st


def session_manager(session_objects: dict):
    for key, value in session_objects.items():
        st.session_state[key] = value


def log_session_var(vars: tuple):
    for var in vars:
        name = [k for k, v in globals().items() if v is var][0]
        st.session_state[name] = var


def _test_single(X_test, Y_test, image_index):
    image = X_test[image_index]  # Get the corresponding image data from X_test
    label = Y_test[image_index]  # Get the corresponding label data from y_test

    image = image.reshape(28, 28)  # Reshape the image array from (28, 28, 1) to (28, 28)

    plt.imshow(image, cmap='gray')  # Display the image using grayscale color map
    plt.title(f"Label: {np.argmax(label)}")  # Set the title of the plot with the true label
    plt.axis('off')  # Remove the axis labels

    return plt.gcf()  # Return the current figure for later plotting

# Set up streamlit session
session_objects = {
    'X_test': None,
    'X_train': None,
    'X_val': None,
    'Y_test': None,
    'Y_train': None,
    'Y_val': None,
    'model': None,
    'history': None
}
session_manager(session_objects)

# run preprocess to set up data 
st.markdown('''
    ## Run Pre-processing
    In this section we ingest the MNIST dataset sourced from the 
    following [kaggle dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download).
    The data is pre-divided into training and test datasets. 
    We vectorise the input data and assign it to the variable `X`. 
    We apply one-hot encoding to the labels and assign it the value `Y`.
    The training and test datasets are then returned as `X_train`, `X_test`, `X_val`, `Y_train`, `Y_test`, and `Y_val`,.
''')
if st.button("Run Pre-processing"):
    X_test, X_train, X_val, Y_test, Y_train, Y_val = pp.pre_process()
    log_session_var((X_test, X_train, X_val, Y_test, Y_train, Y_val))

# configure model 
st.markdown('''
    ## Build convolutional model
    We now build the model keras and tensorflow
''')
if st.button("Build convolutional model"):
    model = cp.compile_model()
    log_session_var((model))

# train model 
st.markdown('''
    ## Train model on training dataset
    We train the model on the training datasets, 15 epochs trained over a batch size of 512.
    We record the history of the loss function, and model accuracy for use in the next step.
''')
if st.button("Train model on training dataset"):
    history = tr.train_model(st.session_state[X_train], st.session_state[Y_train],
                    validation_data=(st.session_state[X_val], st.session_state[Y_val]),
                    epochs=15,
                    batch_size=512)
    log_session_var((history))

# train model 
st.markdown('''
    ## Testing model
    Model has been trained, here we'll compare the predictions of the model to individual images just 
    for demonstration purposes.
''')
if st.button("Test model"):
    
    index = random.randint(0, len(st.session_state[X_test])-1)  # Generate a random index within the range of the test set
    label = st.session_state[Y_test][index]
    figure = _test_single(st.session_state[X_test], st.session_state[Y_test], index)

    prediction = model.predict(np.expand_dims(st.session_state[X_test][index], axis=0))
    predicted_label = np.argmax(prediction)

    st.write("Prediction:", predicted_label)
    st.write("True Label:", np.argmax(label))
    
    plt.show()  # Show the plot


# analyse training data and test model
st.write('## Test model on test dataset')
if st.button("Analyse test data"):
    test_loss, test_acc = ts.analyse_history(model, history, st.session_state[X_test], st.session_state[Y_test])