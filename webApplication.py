import streamlit as st
#import tensorflow as tf
from PIL import Image
#import numpy as np
#import cv2


# Load the trained model
#def load_model():
#    model = tf.keras.models.load_model('model.hdf5')
#    return model
#model = load_model()

st.set_page_config(
    page_title="Skin Lesion Classifier App",
    page_icon="🌟",
    layout="centered",  # You can use "wide" or "centered" layout
    initial_sidebar_state="collapsed",  # Collapsed the sidebar on initial load
)
# Add your logo and resize it
logo_path = "ml-test-skinClassification/logo.JPG"  # Replace with the path to your logo
logo = Image.open(logo_path)
logo.thumbnail((100, 100))  # Adjust the size as needed

## Create a layout with two columns
#col1, col2 = st.columns([1, 4])

## Add the resized logo to the first column
#col1.image(logo, use_column_width=True)

## Add a horizontal rule to separate the logo from the title
#col1.write('<hr style="height:2px; width:530%; background-color: black">', unsafe_allow_html=True)

## Add the title to the second column
#col2.title("Skin Lesion Classifier App")

## Sidebar for image upload and camera capture
#uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#capture_image = st.sidebar.button("Capture Image")

#def preprocess_image(image, target_size):
#    img = Image.open(image).resize(target_size)
#    img_array = np.asarray(img)
#    if img_array.shape[-1] == 4:  # Fix here, use img_array instead of img
#        img_array = img_array[:, :, :3]
#    img_array = img_array / 255.0
#    img_array = np.expand_dims(img_array, axis=0)
    
#    return img_array

## Check if an image is uploaded
#if uploaded_image is not None:
#    # Display the uploaded image
#    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#    # Perform image classification using your model
#    # Preprocess the image (resize, normalize, etc.) as needed
#    img_array = preprocess_image(uploaded_image, target_size=(64, 64))
    
#    # Make predictions
#    predictions = model.predict(img_array)

#    # Get the class with the highest probability
#    predicted_class = np.argmax(predictions)

#    if (predicted_class == 1):
#        st.write(f"Diagnosis: Normal Skin Lesion/ Benign")
#    elif (predicted_class == 2):
#        st.write(f"Diagnosis: Benign Keratosis-like Lesion (BKL)")
#    elif (predicted_class == 3):
#        st.write(f"Diagnosis: Dermatofibroma (DF)")
#    elif (predicted_class == 4):
#        st.write(f"Diagnosis: Melanoma (MEL)")
#    elif (predicted_class == 5):
#        st.write(f"Diagnosis: Melanocytic Nevi (NV)")
#    elif (predicted_class == 6):
#        st.write(f"Diagnosis: Vascular Lesion (VASC)")
#    elif (predicted_class == 0):
#        st.write(f"Diagnosis: Actinic Keratoses (AKIEC)")

#if capture_image:
#    # Add a delay before capturing the image (you can adjust the duration)
#    time.sleep(2)  # 2 seconds delay

#    # Use OpenCV to capture an image from the camera
#    cap = cv2.VideoCapture(0)  # 0 represents the default camera
#    ret, frame = cap.read()
#    cap.release()

#    # Convert OpenCV frame to Pillow image
#    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#    # Display the captured image
#    st.image(pil_image, caption="Captured Image", use_column_width=True)

#    # Perform image classification using your model
#    # Preprocess the image (resize, normalize, etc.) as needed
#    img_array = preprocess_image(frame, target_size=(64, 64))
    
#    # Make predictions
#    predictions = model.predict(img_array)

#    # Get the class with the highest probability
#    predicted_class = np.argmax(predictions)

#    if (predicted_class == 1):
#        st.write(f"Diagnosis: Normal Skin Lesion/ Benign")
#    elif (predicted_class == 2):
#        st.write(f"Diagnosis: Benign Keratosis-like Lesion (BKL)")
#    elif (predicted_class == 3):
#        st.write(f"Diagnosis: Dermatofibroma (DF)")
#    elif (predicted_class == 4):
#        st.write(f"Diagnosis: Melanoma (MEL)")
#    elif (predicted_class == 5):
#        st.write(f"Diagnosis: Melanocytic Nevi (NV)")
#    elif (predicted_class == 6):
#        st.write(f"Diagnosis: Vascular Lesion (VASC)")
#    elif (predicted_class == 0):
#        st.write(f"Diagnosis: Actinic Keratoses (AKIEC)")

# Add a footer with additional information
st.text("This is a simple skin lesion classifier web app.")
