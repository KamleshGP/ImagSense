# Image Sense 📸

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://image-sense.streamlit.app/)

**Welcome to Image Sense!** This web application provides a user-friendly way to classify images of various natural and man-made scenes. Simply upload an image, and our intelligent system will predict whether it belongs to the categories of Dew, Forest, Glacier, Mountain, or Plastic Bottles. Built with Streamlit and powered by a custom-trained image recognition model based on the VGG16 architecture, Image Sense offers a glimpse into the world of image classification.

## ⚙️ Setup and Installation (for local development)

If you want to run this application locally, follow these steps:

1.  **Clone the repository** (if you have the code in a repository):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required libraries**:
    ```bash
    pip install streamlit tensorflow keras numpy pandas Pillow
    ```
    It's recommended to use a virtual environment to manage dependencies.

3.  **Ensure the model file is present**: Make sure the `image_reco_vgg.keras` file is in the same directory as your Streamlit script or in a location accessible by the script. This file contains the weights of your custom-trained VGG16-based model.

4.  **Run the Streamlit application**:
    ```bash
    streamlit run your_script_name.py
    ```
    Replace `your_script_name.py` with the actual name of your Python script (e.g., `app.py`).

## 🛠️ How to Use

1.  Go to the deployed Streamlit application: [https://image-sense.streamlit.app/](https://image-sense.streamlit.app/)
2.  Click on the "Browse files" button to upload an image from your local device.
3.  Once the image is uploaded, it will be displayed on the screen.
4.  Click the "Classify" button to get the prediction.
5.  The predicted class of the image will be displayed below the button.

## Under the Hood

This application utilizes a custom-trained image classification model built upon the **VGG16 architecture**. Here's a breakdown of the process:

1.  **Base Model**: The VGG16 model, a deep convolutional neural network known for its strong performance in image recognition, was used as the foundation. This pre-trained model learned hierarchical feature representations from a massive dataset (like ImageNet).
2.  **Customization**: Instead of using the full pre-trained VGG16 for a thousand classes, the final classification layers were replaced and fine-tuned (or the entire model might have been trained) specifically for the five classes relevant to this application: Dew, Forest, Glacier, Mountain, and Plastic Bottles. This process allows the model to leverage the general image features learned by VGG16 while specializing in our specific classification task.
3.  **Image Processing**: When you upload an image, it's preprocessed to match the input size expected by the VGG16 model (224x224 pixels). The pixel values are also typically scaled or normalized.
4.  **Prediction**: The preprocessed image is passed through the custom-trained VGG16-based model. The model outputs a probability distribution across the five classes.
5.  **Result**: The class with the highest probability is selected as the predicted label for the uploaded image.

The `image_reco_vgg.keras` file you provided contains the saved weights of this custom-trained VGG16 model.

## Acknowledgement

This application utilizes the following open-source libraries:

-   **Streamlit**: For creating the interactive web application.
-   **TensorFlow** and **Keras**: For building, loading, and using the custom-trained VGG16-based machine learning model.
-   **NumPy**: For efficient numerical computations.
-   **Pandas**: For data manipulation, particularly in handling the model's output.
-   **Pillow (PIL)**: For image processing tasks.

The base VGG16 architecture was developed by the Visual Geometry Group at the University of Oxford. We acknowledge their significant contribution to the field of computer vision. The custom training of this model was performed by Kamlesh Prajapati.
