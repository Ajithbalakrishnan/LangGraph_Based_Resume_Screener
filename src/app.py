import streamlit as st


def main():
    st.title("My Streamlit App")
    st.text("This is some text.")
    
    st.write("Hello, ","name")
    name = st.text_input("Enter your name:")
    if st.button("Click me"):
        st.write("You clicked the button!")
        
    age = st.slider("Select your age", 0, 100)
    
    agree = st.checkbox("I agree to the terms and conditions")
    
    options = ["Option 1", "Option 2", "Option 3"]
    selected_option = st.selectbox("Choose an option:", options)
    
    uploaded_file = st.file_uploader("Upload a file")

if __name__ == "__main__":
    main()