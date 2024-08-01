# AIPI540-Pick4U

## Description
This application helps to recommend the best fruit by analysing through a photo.

## Author
Xinyue(Yancey) Yang

## Model
**Naive**: Object dection by template matching, random recommendation.

**ML**: Object dection by contours finding, recommendation based on color-hist.

**Neural Network**: Object detection by YOLO-v8, recommendation based on ResNet50. 

## Instruction
To run the code in this repository, please follow these steps:

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/Talos6/AIPI540-Pick4U.git
    ```

2. Navigate to the project directory:
    ```
    cd AIPI540-Pick4U
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the application:
    ```
    streamlit run main.py
    ```

5. Re-train the model:
    ```
    python setup.py