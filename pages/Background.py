import streamlit as st


st.header('Convolutional Neural Networks')
# def show_cnn():
st.markdown("""
Convolutional Neural Networks (CNNs) are a type of deep learning algorithm that have gained
significant popularity and achieved remarkable success in various computer vision tasks, such
as image classification, object detection, and image segmentation.\n
CNNs are inspired by how the visual cortex of animals, particularly mammals, processes visual
information. They are designed to automatically learn and extract meaningful features from 
input data, primarily images, by leveraging the concept of convolution.

The key components of a CNN are:

- Convolutional Layers: These layers consist of multiple filters (also called kernels) that slide
   over the input image, performing element-wise multiplications and additions. The purpose is to extract 
   local features by detecting patterns, edges, textures, or other specific characteristics. 
   Convolutional layers help capture both low-level and high-level features.
- Pooling Layers: After each convolutional layer, a pooling layer is often added to reduce the 
   spatial dimensions of the feature maps. Pooling operations, such as max pooling or average pooling, 
   aggregate information from local regions, reducing the computational complexity and providing 
   robustness against small spatial variations.
- Activation Functions: Non-linear activation functions like ReLU (Rectified Linear Unit) are
   typically applied after each convolutional and pooling layer. Activation functions introduce 
   non-linearity, allowing the CNN to model complex relationships between the input data.
- Fully Connected Layers: Towards the end of the network, fully connected layers are used to 
   perform classification or regression on the learned features. These layers connect every neuron
   from the previous layer to each neuron in the next layer, enabling the network to make predictions
   based on the extracted features.

Training a CNN involves using labeled training data to adjust the weights of the network through a 
process called backpropagation. This optimization technique minimizes the difference between predicted
outputs and actual labels using gradient descent or its variations.\n
One of the major strengths of CNNs lies in their ability to capture hierarchical spatial dependencies 
in images and automatically learn relevant features, reducing the need for manual feature engineering.
This makes CNNs particularly effective in tasks where understanding the spatial relationships within 
an image is crucial.\n
With their impressive performance and versatility, CNNs have revolutionized computer vision applications
and become a cornerstone of deep learning techniques in many fields.""")
