from scripts.nn_approach import NNApproach
import streamlit as st
from PIL import Image, ImageOps

def run():
    # Side Panel
    st.sidebar.title("Pick4U")

    # Content
    page = st.sidebar.radio('Go To', ["Application", "About", "Details"])

    if page == "Application":
        # Application
        nn_approach = NNApproach()

        # Elements
        default_image = Image.open("data/streamlit/default.png")
        image_display = st.image(default_image, use_column_width=True)
        rec_config = st.slider("Number of Picks", 1, 20, 3)
        uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

        # Actions
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            grey_image = ImageOps.grayscale(image)

            # Spinner for processing
            with st.spinner('Uploading... Detecting... Evluating... Ranking... Recomending... Drawing... Responding...'):
                image_display.image(grey_image, use_column_width=True)
                result = nn_approach.process(image, rec_config)
            image_display.image(result, use_column_width=True)

    elif page == "About":
        # About
        st.write("### Intro")
        st.write("Pick4U is a web application that allows users to take a photo of fruits and receive a selection of the products based on their quality. It mainly targets the daily users that are facing a hard situation of purchasing fruits in the marketplace. The application will highlight which items are best for purchasing based on visual cues.")
        st.write("### Problem")
        st.write("Consumers often struggle to determine the quality of fruits at the point of purchase. Misjudging the quality can lead to purchasing subpar produce, resulting in wasted money and resources.")
        st.write("### Initiative")
        st.write("This idea was inspired by a recent activity after GPT4o launched. Some users are asking GPT4o to select a watermelon to purchase. This has become a very popular topic in TikTok and a close interaction of daily life with AI.")
        st.write("### Solution")
        st.write("Pick4U aims to provide a easy and quick solution to help users to select the best fruits in the market. Enhacing the user daily life with AI.")
        st.write("### Uniqueness")
        st.markdown("""
        - **Accessibility**: Web application that can be accessible any time, any where and instant response time.
        - **Fine-grained Object Recognition**: Accurately recognizing multiple and similar fruit units in a single picture.
        - **Visual Cues**: Scoring based on various image features and deliver a more accurate result compared to LLM models.
        """)
        st.write("### Spike")
        st.markdown("""
        - **Front End**: An interactive web application that allows users to upload the photo and receive the recommendation.
        - **Object Detection**: Fruit unit detection by model with learned features.
        - **Quality Scoring**: Quality determined by visual cues.
        - **Recommendation**: Picking the best items based on the quality score, response with indicator. 
        """)
        st.write("### Usage")
        st.markdown("""
        - Upload a photo taken from grocery store.
        - Number of picks can be configured through the slide bar.
        - Check the responses with highlighted items, go for it. 
        """)
        st.write("### Author")
        st.write("Xinyue(Yancey) Yang")
        st.write("### Repository")
        st.write("[Github link](https://github.com/Talos6/AIPI540-games)")
        st.write("### Instructions")
        code = """
            # Clone repo and ensure python and pip have installed
            git clone https://github.com/Talos6/AIPI540-Pick4U.git 

            # Install required libraries
            pip install -r requirements.txt

            # Run the application with trained model
            streamlit run main.py

            # Run the script to re-setup models
            python setup.py
        """
        st.code(code, language='bash')

    elif page == "Details":
        # Details
        st.write("### Source")
        st.write("The data source is collected from Kaggle datasets.")
        st.markdown("""
        - **Fruit Unit Recognition**: Kaggle Fruits 100 dataset with 100 classes of labeled fruits and vegetables. [Kaggle Link](https://www.kaggle.com/datasets/marquis03/fruits-100)
        - **Quality Scoring**: Kaggle Fruits fresh and rotten for classification dataset with fresh and rotten images of apple, banana and organge. [Kaggle Link](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
        """)
        st.write("### Naive Approach")
        st.markdown("""
        - **Data Pipeline**: Image data loaded through cv2.imread.
        - **Detection & Recognition**: Random sampling input data and prepare templates, for detection and recognition.
        - **Scoring & Recommendation**: All recognized units has the same score and recommendation based on randome selection.
        - **Evaluation**: Accuracy(Detection) = 23%, Accuracy(Recognition) = 81%, Accuracy(Scoring) = N/A, Satisfaction(Recommendation) = 0%.
        - **Conclusion**: The least viable solution, it only represents the successful condition of the workflow.
        """)
        st.write("### Machine Learning Approach")
        st.markdown("""
        - **Data Pipeline**: Image data loaded and prepared color-hist features.
        - **Detection**: Fruit units detection done by contour finding.
        - **Recognition**: A trained SVM model with learned color-hist features and classified fruits unit with labels.
        - **Scoring & Recommendation**: A trained NB model perform binary classification where represents probability as score. Recommendation based on the highest score.
        - **Evaluation**: Accuracy(Detection) = 58%, Accuracy(Recognition) = 95%, Accuracy(Scoring) = 93%, Satisfaction(Recommendation) = 40%.
        - **Conclusion**: The ml models shows a strong power in classification but still has a gap in fine-grained object recognition.
        """)
        st.write("### Neural Network Approach")
        st.markdown("""
        - **Data Pipeline**: Image data converted to torch dataset with transforms.
        - **Detection & Recognition**: Fruit units detection and recognition done by pre-trained YOLO-v8n models with guards to specific fruit types.
        - **Scoring & Recommendation**: Transfer learning with pre-trained ResNet-50 model to score the fruits quality through health/rotten. Recommendation based on the highest score.
        - **Evaluation**: Accuracy(Detection) = 83%, Accuracy(Recognition) = 96%, Accuracy(Scoring) = 98%, Satisfaction(Recommendation) = 80%.
        - **Conclusion**: Approach in SOTA state, with high accuracy and satisfaction rate.
        """)
        st.write("### Prior Efforts")
        st.markdown("""
        - **LLM model**: Interaction with GPT4o shows a diversed feedback, Satisfaction(Recommendation) = 60%.
        - **Classification**: Datasets are all aimed for classification problem where majority of work's accuracy around 95%.
        """)
        st.write("### Future Work")
        st.markdown("""
        - **Data Collection**: More relevant data needs to be collected and better labeled.
        - **Fine-grained detection**: Can apply Fine-grained object detection for better segmentation.
        - **Feature Engineering**: More features can be extracted or prepared for better heuristic.
        """)

if __name__ == "__main__":
    run()