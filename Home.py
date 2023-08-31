import streamlit as st
from pages import Background, Dataset, MNIST_page

def main():
    st.set_page_config(
        page_title="CXR",
        page_icon="🦴",
    )

    st.write("# Welcome to Streamlit! 👋")

    pages = {
        'Background': Background.Background_page,
        'Dataset': Dataset.Dataset_page,
        'MNIST_page': MNIST_page.MNIST_page
    }

    # Create a sidebar with buttons to select the page
    page = st.sidebar.selectbox("Select a page", tuple(pages.keys()))

    # Call the selected page function to display its content
    pages[page]()

if __name__ == "__main__":
    main()