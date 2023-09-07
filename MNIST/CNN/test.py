import matplotlib.pyplot as plt
import streamlit as st


def analyse_history(model, history, X_test, Y_test):  

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    
    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot the model accuracy vs Epochs
    ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
    ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax[0].set_title('Training & Validation Accuracy', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Accuracy', fontsize=16)
    ax[0].legend()
    
    # Plot the loss vs Epochs
    ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
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