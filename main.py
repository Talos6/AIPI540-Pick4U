from scripts.naive_approach import NaiveApproach
from scripts.ml_approach import MLApproach
from scripts.nn_approach import NNApproach
import streamlit as st
from PIL import Image

def run():
    nn_approach = NNApproach()
    
    st.title("Pick4U")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_display = st.image(image, use_column_width=True)

        with st.spinner('Processing...'):
            result = nn_approach.process(image, 3)
        image_display.image(result, use_column_width=True)

if __name__ == "__main__":
    run()