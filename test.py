import streamlit as st

def session_manager(session_objects: dict):
    for key, value in session_objects.items():
        st.session_state[key] = value

def log_session_var(vars: tuple):
    for var in vars:
        name = [k for k, v in globals().items() if v is var][0]
        st.session_state[name] = var

# Set up streamlit session
session_objects = {
    'X_test': None,
    'X_train': None,
    'X_val': None,
}
session_manager(session_objects)

# Test logging session variables
X_test = [1, 2, 3]
X_train = [4, 5, 6]
X_val = [7, 8, 9]

log_session_var((X_test, X_train, X_val))

# Print session state
for key, value in st.session_state.items():
    st.write(key, "=", value)
