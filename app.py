



import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import os
from datetime import datetime
import json

# Create a directory to store captured images and metadata
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

METADATA_FILE = 'metadata.json'

# Load existing metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as file:
            return json.load(file)
    return []

# Save metadata to file
def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as file:
        json.dump(metadata, file)

metadata = load_metadata()

# Function to compute the histogram of a face region
def compute_face_histogram(image, face_region):
    x, y, w, h = face_region
    face = image[y:y+h, x:x+w]
    face_hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    face_hist = cv2.normalize(face_hist, face_hist).flatten()
    return face_hist

# Load saved face histograms
saved_face_histograms = []
for entry in metadata:
    img = cv2.imread(entry['file'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        face_hist = compute_face_histogram(img, faces[0])
        saved_face_histograms.append((face_hist, entry['name'], entry['number']))

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.capture_image = False
        self.captured_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and recognize them
        for (x, y, w, h) in faces:
            face_hist = compute_face_histogram(frm, (x, y, w, h))

            name = "Unknown"
            number = ""
            best_match_score = float('inf')
            for saved_face_hist, saved_name, saved_number in saved_face_histograms:
                score = cv2.compareHist(face_hist, saved_face_hist, cv2.HISTCMP_CORREL)
                if score < best_match_score:
                    best_match_score = score
                    name = saved_name
                    number = saved_number

            if best_match_score < 0.6:  # Adjust threshold based on your requirements
                cv2.putText(frm, f"{name} - {number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # If capture button was clicked, store the frame with faces
        if self.capture_image:
            self.capture_image = False
            self.captured_frame = frm

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

    def get_captured_frame(self):
        return self.captured_frame

def save_frame(frame, name, number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_images/{name}_{number}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def display_saved_images():
    for entry in metadata:
        st.image(entry['file'], caption=f"{entry['name']} - {entry['number']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Delete", key=f"delete_{entry['file']}"):
                if os.path.exists(entry['file']):
                    os.remove(entry['file'])
                metadata.remove(entry)
                save_metadata(metadata)
                st.experimental_rerun()
        with col2:
            if st.button("Edit", key=f"edit_{entry['file']}"):
                st.session_state['editing'] = entry

        # Editing section
        if 'editing' in st.session_state and st.session_state['editing'] == entry:
            st.text_input("Edit Name", value=entry['name'], key=f"edit_name_{entry['file']}")
            st.text_input("Edit Number", value=entry['number'], key=f"edit_number_{entry['file']}")
            if st.button("Save Changes", key=f"save_changes_{entry['file']}"):
                entry['name'] = st.session_state[f"edit_name_{entry['file']}"]
                entry['number'] = st.session_state[f"edit_number_{entry['file']}"]
                save_metadata(metadata)
                del st.session_state['editing']
                st.experimental_rerun()
        with col3:
            pass

def main():
    st.title("Face Detection and Capture")

    # Initialize session state variables
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
    if 'number' not in st.session_state:
        st.session_state['number'] = ""

    ctx = webrtc_streamer(
        key="key",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        if st.button("Capture Image"):
            ctx.video_processor.capture_image = True

        captured_frame = ctx.video_processor.get_captured_frame()
        if captured_frame is not None:
            st.image(captured_frame, caption="Captured Image")

            st.session_state['name'] = st.text_input("Enter Name", st.session_state['name'])
            st.session_state['number'] = st.text_input("Enter Number", st.session_state['number'])

            if st.button("Save Image"):
                filename = save_frame(captured_frame, st.session_state['name'], st.session_state['number'])
                metadata.append({
                    'name': st.session_state['name'],
                    'number': st.session_state['number'],
                    'file': filename
                })
                save_metadata(metadata)
                st.success(f"Image saved as {filename}")

    st.header("Saved Images")
    display_saved_images()

if __name__ == "__main__":
    main()

