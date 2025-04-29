import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="😊 Emotion Detector", page_icon="😎", layout="centered")

import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import re


# Load pre-trained model
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_recognition_model.h5")


model = load_emotion_model()

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Initialize session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


# Function to detect emotion
def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    prediction = model.predict(reshaped, verbose=0)
    emotion_idx = np.argmax(prediction)
    return emotion_labels[emotion_idx]


# Login Page
if not st.session_state.logged_in:
    st.subheader("🔐 Sign In")

    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login"):
        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        special_characters = r"[!@#$%^&*(),.?\":{}|<>]"

        valid_email = re.match(email_pattern, email)
        valid_password = len(password) >= 8 and re.search(special_characters, password)

        if not valid_email:
            st.error("❌ Please enter a valid email address.")
        elif not valid_password:
            st.error(
                "❌ Password must be at least 8 characters long and contain a special character."
            )
        else:
            st.success("✅ Login successful!")
            st.session_state.logged_in = True

# Main App
if st.session_state.logged_in:
    st.sidebar.title("🔎 Navigation")
    page = st.sidebar.radio(
        "✨ Choose a page:", ["🏠 Home", "📸 Camera", "🖼️ Upload Image", "ℹ️ About"]
    )

    if page == "🏠 Home":
        st.subheader("🏠 Welcome to the Emotion Detector App!")

        st.markdown(
            """
        This app helps you identify emotions through **facial expressions** using deep learning models.
        
        🎯 **Features:**
        - **Real-time emotion detection** via your webcam 📸
        - **Upload an image** and detect emotions 🖼️
        - **AI-powered** using a deep learning model built with **TensorFlow** and **Keras** 🚀
        
        💡 Simply log in and start exploring:
        - **Camera**: Use your webcam for live emotion detection
        - **Upload Image**: Upload an image for analysis
        - **About**: Learn more about the app

        Let's get started!
        """
        )

        st.success("👉 Use the Sidebar to navigate to different features!")

    elif page == "📸 Camera":
        st.subheader("📷 Real-Time Emotion Detection")

        start = st.button("▶️ Start Camera")
        stop = st.button("⏹️ Stop Camera")

        if start:
            st.session_state.camera_running = True
        if stop:
            st.session_state.camera_running = False

        FRAME_WINDOW = st.empty()
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        camera = cv2.VideoCapture(0)

        while st.session_state.camera_running:
            ret, frame = camera.read()
            if not ret:
                st.error("Camera error. Please try again.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for x, y, w, h in faces:
                face_img = frame[y : y + h, x : x + w]
                emotion = detect_emotion(face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        camera.release()
        cv2.destroyAllWindows()

    elif page == "🖼️ Upload Image":
        st.subheader("🖼️ Upload an Image to Detect Emotion")

        uploaded_file = st.file_uploader(
            "📂 Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing the Image..."):
                img = Image.open(uploaded_file)
                img_array = np.array(img)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(img_array, caption="Uploaded Image", width=300)

                if img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array

                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) == 0:
                    st.warning("⚠️ No face detected.")
                else:
                    x, y, w, h = faces[0]
                    face_img = img_bgr[y : y + h, x : x + w]
                    detected_emotion = detect_emotion(face_img)

                    st.markdown(
                        f"<h3 style='text-align: center;'>😎 Detected Emotion: <span style='color: orange;'>{detected_emotion}</span></h3>",
                        unsafe_allow_html=True,
                    )

    if page == "ℹ️ About":
        st.subheader("ℹ️ About This App")

        st.markdown(
            """
        **Emotion Detector App** helps you recognize emotions based on facial expressions using **AI** and **Deep Learning**.
        
        **Built with:**
        - Python 🐍
        - TensorFlow & Keras 🧠 for emotion detection
        - OpenCV 📷 for real-time image processing
        - Streamlit 🎈 for the user interface
        
        🚀 **How it works:**
        - The app uses a pre-trained **Convolutional Neural Network (CNN)** model to predict the emotion.
        - It processes facial expressions and classifies them into one of the 7 emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.
        
        **Made with ❤️ by:**
        - **Snehashis Das**
        - **Avijit Paul**

        This project aims to demonstrate how deep learning can be applied to real-world problems, such as emotion recognition.
        """
        )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.camera_running = False
        st.success("🔓 Logged out successfully!")
