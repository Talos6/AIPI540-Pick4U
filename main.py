from scripts.nn_approach import NNApproach
import streamlit as st
from PIL import Image, ImageOps

def run():
    st.sidebar.title("Pick4U")
    page = st.sidebar.radio('Go To', ["Application", "About"])

    if page == "Application":
        nn_approach = NNApproach()
    
        default_image = Image.open("data/streamlit/default.png")
        image_display = st.image(default_image, use_column_width=True)
        rec_config = st.slider("Number of Picks", 1, 20, 3)
        uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            grey_image = ImageOps.grayscale(image)

            with st.spinner('Uploading... Detecting... Evluating... Ranking... Recomending... Drawing... Responding...'):
                image_display.image(grey_image, use_column_width=True)
                result = nn_approach.process(image, rec_config)
            image_display.image(result, use_column_width=True)

    elif page == "About":
        st.write("## Reference")
        st.write("### Author")
        st.write("Xinyue(Yancey) Yang")
        st.write("### Repository")
        st.write("[Github link](https://github.com/Talos6/AIPI540-games)")
        st.write("### Instructions")
        code = """
            # Clone repo and ensure python and pip have installed
            git clone <repo_link> 

            # Install required libraries
            pip install -r requirements.txt

            # Run the application with trained model
            streamlit run main.py

            # Run the script to re-setup models
            python setup.py
        """
        st.code(code, language='bash')

if __name__ == "__main__":
    run()