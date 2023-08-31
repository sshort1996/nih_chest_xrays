import streamlit as st


def train_model(X_train,
                Y_train, 
                model,
                X_val, 
                Y_val):
    
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=15,
                        batch_size=512)
    
    return history