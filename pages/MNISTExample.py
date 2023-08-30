import streamlit as st
from MNIST.Preprocess import import_data, prep_data, plot_data, vectorise_data
from MNIST.compile import compile
import matplotlib.pyplot as plt

st.write("# MNIST handwritten digits")
x_train, y_train, x_test, y_test, inpx, img_cols, img_rows = import_data()
x_train, y_train, x_test, y_test = prep_data(x_train, y_train, x_test, y_test)
fig = plot_data(x_train, y_train, img_rows, img_cols)
y_train, y_test = vectorise_data(y_train, y_test) 

st.write("""
First we work on importing and preprocessing the MNIST dataset, as well as plotting a sample of the training data. These functions are useful for implementing a Convolutional Neural Network (CNN) approach to solving the MNIST handwritten digits recognition problem.

Here's a high-level summary of what the code in this section achieves:

1. Importing the data: The code uses Keras to load the MNIST dataset, which consists of images of handwritten digits along with their corresponding labels. The data is then reshaped based on the image_data_format to prepare it for input into a CNN.

2. Preprocessing the data: The code preprocesses the input data by converting the pixel values to float32 and scaling them between 0 and 1. It also converts the target labels into one-hot encoded vectors. This preprocessing step is important for preparing the data to be fed into a CNN model.

3. Plotting a sample of the training data: The code includes a function that plots a sample of the training data. This is useful for visualizing the images and their corresponding labels, which can help in understanding the dataset and verifying the correctness of the data preprocessing.

Below is a sample of the MNIST data shown alongside it's annotations
""")

# Display the plot using Streamlit
st.write("## Sample of MNIST training dataset")
st.pyplot(fig)

st.write("## Building the CNN model")
compile()