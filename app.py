import streamlit as st

def main():
    # Set the title of the web app
    st.title("Final Project NLP")

    # Display the names of the students
    st.header("Project Team Members:")
    st.write("1. Jiwoo Suh")
    st.write("2. Sanjana")

    # Section for Upmanyu
    st.header("3. Upmanyu")
    st.write("Details of different models:")

    # Model 1 Details
    with st.expander("Model 1"):
        st.write("Details about Model 1...")

    # Model 2 Details
    with st.expander("Model 2"):
        st.write("Details about Model 2...")

    # Model 3 Details
    with st.expander("Model 3"):
        st.write("Details about Model 3...")

if __name__ == "__main__":
    main()
