

import logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import numpy as np
import os
import cv2
from datetime import datetime
import json
import threading

logging.basicConfig(level=logging.DEBUG)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:YOUR_TURN_SERVER"], "username": "USERNAME", "credential": "PASSWORD"}
    ]
})

if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

METADATA_FILE = 'metadata.json'

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as file:
            return json.load(file)
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as file:
        json.dump(metadata, file)

metadata = load_metadata()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.capture_image = False
        self.captured_frame = None
        self.frame_skip = 5  # Process every 5th frame
        self.frame_count = 0
        self.lock = threading.Lock()

        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_encodings = []
        self.known_face_metadata = []

        # Load known faces from metadata
        for idx, entry in enumerate(metadata):
            image = cv2.imread(entry['file'], cv2.IMREAD_GRAYSCALE)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_image = image[y:y + h, x:x + w]
                self.known_face_encodings.append((face_image, idx))
                self.known_face_metadata.append({
                    'name': entry['name'],
                    'email': entry['email'],
                    'timestamp': entry['timestamp']
                })
            else:
                logging.warning(f"No face found in {entry['file']}")

        if len(self.known_face_encodings) > 0:
            self.train_recognizer()

    def train_recognizer(self):
        faces = [encoding[0] for encoding in self.known_face_encodings]
        labels = [encoding[1] for encoding in self.known_face_encodings]
        self.face_recognizer.train(faces, np.array(labels))

    def process_faces(self, gray_frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_info = []

        for (x, y, w, h) in faces:
            face_image = gray_frame[y:y + h, x:x + w]
            label, confidence = self.face_recognizer.predict(face_image)

            name = "Unknown"
            email = ""
            if confidence < 100:  # You can adjust this threshold
                metadata_entry = self.known_face_metadata[label]
                name = metadata_entry['name']
                email = metadata_entry['email']

            face_info.append((x, y, w, h, name, email))

        return face_info

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        logging.debug("Frame received for processing")

        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            logging.debug("Skipping frame processing")
            return av.VideoFrame.from_ndarray(frm, format="bgr24")

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # Non-blocking face processing
        face_info = []
        thread = threading.Thread(target=lambda: face_info.extend(self.process_faces(gray_frame)))
        thread.start()
        thread.join(timeout=1)  # Join with timeout to prevent blocking

        logging.debug(f"Detected faces: {face_info}")

        for (x, y, w, h, name, email) in face_info:
            # Draw a box around the face
            logging.debug(f"Drawing rectangle at {(x, y), (x + w, y + h)}")
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw a label with the name and email below the face
            label = f"{name}, {email}" if name != "Unknown" else name
            cv2.rectangle(frm, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frm, label, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

        if self.capture_image:
            logging.debug("Capturing image")
            self.capture_image = False
            self.captured_frame = frm.copy()

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

    def get_captured_frame(self):
        with self.lock:
            return self.captured_frame

def save_frame(frame, name, email):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"captured_images/{name}_{email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)

    metadata.append({
        'name': name,
        'email': email,
        'file': filename,
        'timestamp': timestamp
    })
    save_metadata(metadata)
    return filename

def display_contact_report():
    st.title("History")
    st.write("### Registered Users")
    for entry in metadata:
        st.write(f"Name: {entry['name']}, Email: {entry['email']}, Date & Time: {entry['timestamp']}")

def main():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Live Face", "Register Face", "History"])

    if menu == "Live Face":
        st.title("Live Face Recognition")

        ctx = webrtc_streamer(
            key="live_face",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    elif menu == "Register Face":
        st.title("Register Face")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            st.image(image, channels="BGR")
            name = st.text_input("Name")
            email = st.text_input("Email")
            if st.button("Save"):
                save_frame(image, name, email)
                st.success("Face Registered Successfully")

        st.write("OR")
        st.write("Use the live camera to capture your face")
        
        ctx = webrtc_streamer(
            key="register_face",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            if st.button("Capture Image"):
                ctx.video_processor.capture_image = True

            captured_frame = ctx.video_processor.get_captured_frame()
            if captured_frame is not None:
                st.image(captured_frame, channels="BGR")
                name = st.text_input("Name")
                email = st.text_input("Email")
                if st.button("Save Captured Image"):
                    save_frame(captured_frame, name, email)
                    st.success("Face Registered Successfully")

    elif menu == "History":
        display_contact_report()

if __name__ == "__main__":
    main()





