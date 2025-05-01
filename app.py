import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import re
import time

# Set Streamlit page config
st.set_page_config(page_title="😊 Emotion Detector", page_icon="😎", layout="centered")


# Load model once using cache
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_recognition_model.h5")


model = load_emotion_model()

# Labels for prediction
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Session states
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("camera_running", False)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Helper: Detect emotion from a face image
def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    prediction = model.predict(reshaped, verbose=0)
    return emotion_labels[np.argmax(prediction)]


# Login Page
def login_page():
    st.subheader("🔐 Sign In")
    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login"):
        valid_email = re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)
        valid_password = len(password) >= 8 and re.search(
            r"[!@#$%^&*(),.?\":{}|<>]", password
        )
        if not valid_email:
            st.error("❌ Enter a valid email.")
        elif not valid_password:
            st.error("❌ Password must be 8+ characters and include a special symbol.")
        else:
            st.success("✅ Logged in!")
            st.session_state.logged_in = True


# Home Page
def home_page():
    st.subheader("🏠 Welcome to the Emotion Detector App!")
    st.markdown(
        """
    This app helps identify emotions using deep learning on facial expressions.

    **🎯 Features:**
    - Real-time emotion detection via webcam 📸
    - Upload image for analysis 🖼️
    - Powered by TensorFlow/Keras and OpenCV 🚀
    
    👉 Use the Sidebar to start!
    """
    )


# Camera Page
def camera_page():
    st.subheader("📷 Real-Time Emotion Detection")
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Camera"):
        st.session_state.camera_running = True
    if col2.button("⏹️ Stop Camera"):
        st.session_state.camera_running = False

    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    while st.session_state.camera_running:
        start_time = time.time()
        ret, frame = camera.read()
        if not ret:
            st.error("Camera error.")
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

        fps = int(1 / (time.time() - start_time))
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    cv2.destroyAllWindows()


# Image Upload Page
def upload_page():
    st.subheader("🖼️ Upload Image for Emotion Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_array, caption="Uploaded Image", width=300)

        if img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            st.warning("⚠️ No face detected.")
        else:
            for idx, (x, y, w, h) in enumerate(faces):
                face_img = img_bgr[y : y + h, x : x + w]
                emotion = detect_emotion(face_img)
                st.markdown(
                    f"<h5>Face {idx+1}: <span style='color:orange'>{emotion}</span></h5>",
                    unsafe_allow_html=True,
                )


# About Page
def about_page():
    st.subheader("ℹ️ About This App")
    st.markdown(
        """
    This **Emotion Detector App** identifies emotions using facial expressions with AI.

    **Built using:**
    - Python 🐍
    - TensorFlow & Keras 🧠
    - OpenCV 📷
    - Streamlit 🎈

    **Detectable Emotions:**
    - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

    **Made by ❤️**
    - Snehashis Das
    - Avijit Paul
    """
    )


# Main Logic
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("🔎 Navigation")
    page = st.sidebar.radio(
        "✨ Go to:", ["🏠 Home", "📸 Camera", "🖼️ Upload Image", "ℹ️ About"]
    )

    if page == "🏠 Home":
        home_page()
    elif page == "📸 Camera":
        camera_page()
    elif page == "🖼️ Upload Image":
        upload_page()
    elif page == "ℹ️ About":
        about_page()

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.camera_running = False
        st.success("🔓 Logged out successfully!")
