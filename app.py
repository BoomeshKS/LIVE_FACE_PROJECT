
# import logging
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
# import av
# import cv2
# import numpy as np
# import os
# from datetime import datetime
# import json

# logging.basicConfig(level=logging.DEBUG)

# # Ensure that your TURN server credentials are correct
# RTC_CONFIGURATION = RTCConfiguration({
#     "iceServers": [
#         {"urls": ["stun:stun.l.google.com:19302"]},
#         {"urls": ["turn:YOUR_TURN_SERVER"], "username": "USERNAME", "credential": "PASSWORD"}
#     ]
# })

# if not os.path.exists('captured_images'):
#     os.makedirs('captured_images')

# METADATA_FILE = 'metadata.json'

# def load_metadata():
#     if os.path.exists(METADATA_FILE):
#         with open(METADATA_FILE, 'r') as file:
#             return json.load(file)
#     return []

# def save_metadata(metadata):
#     with open(METADATA_FILE, 'w') as file:
#         json.dump(metadata, file)

# metadata = load_metadata()

# def compute_face_histogram(image, face_region):
#     x, y, w, h = face_region
#     face = image[y:y+h, x:x+w]
#     face_hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     face_hist = cv2.normalize(face_hist, face_hist).flatten()
#     return face_hist

# saved_face_histograms = []
# for entry in metadata:
#     img = cv2.imread(entry['file'])
#     if img is None:
#         continue
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     if len(faces) > 0:
#         face_hist = compute_face_histogram(img, faces[0])
#         saved_face_histograms.append((face_hist, entry['name'], entry['number']))

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.capture_image = False
#         self.captured_frame = None
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     def recv(self, frame):
#         try:
#             frm = frame.to_ndarray(format="bgr24")
#             gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
#             faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             for (x, y, w, h) in faces:
#                 face_hist = compute_face_histogram(frm, (x, y, w, h))

#                 name = "Unknown"
#                 number = ""
#                 best_match_score = float('inf')
#                 for saved_face_hist, saved_name, saved_number in saved_face_histograms:
#                     score = cv2.compareHist(face_hist, saved_face_hist, cv2.HISTCMP_CORREL)
#                     if score < best_match_score:
#                         best_match_score = score
#                         name = saved_name
#                         number = saved_number

#                 if best_match_score < 0.6:
#                     cv2.putText(frm, f"{name} - {number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
#                     cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 else:
#                     cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 0, 255), 2)

#             if self.capture_image:
#                 self.capture_image = False
#                 self.captured_frame = frm

#             return av.VideoFrame.from_ndarray(frm, format="bgr24")
#         except Exception as e:
#             logging.error(f"Error in recv: {e}")
#             return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")

#     def get_captured_frame(self):
#         return self.captured_frame

# def save_frame(frame, name, number):
#     try:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"captured_images/{name}_{number}_{timestamp}.jpg"
#         cv2.imwrite(filename, frame)
#         return filename
#     except Exception as e:
#         logging.error(f"Error saving frame: {e}")
#         return None

# def display_saved_images():
#     st.title("Saved Images")
#     for entry in metadata:
#         if entry['file']:
#             st.image(entry['file'], caption=f"{entry['name']} - {entry['number']}")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("Delete", key=f"delete_{entry['file']}"):
#                 if entry['file'] and os.path.exists(entry['file']):
#                     os.remove(entry['file'])
#                 metadata.remove(entry)
#                 save_metadata(metadata)
#                 st.experimental_rerun()
#         with col2:
#             if st.button("Edit", key=f"edit_{entry['file']}"):
#                 st.session_state['editing'] = entry

#         if 'editing' in st.session_state and st.session_state['editing'] == entry:
#             st.text_input("Edit Name", value=entry['name'], key=f"edit_name_{entry['file']}")
#             st.text_input("Edit Number", value=entry['number'], key=f"edit_number_{entry['file']}")
#             if st.button("Save Changes", key=f"save_changes_{entry['file']}"):
#                 entry['name'] = st.session_state[f"edit_name_{entry['file']}"]
#                 entry['number'] = st.session_state[f"edit_number_{entry['file']}"]
#                 save_metadata(metadata)
#                 del st.session_state['editing']
#                 st.experimental_rerun()
#         with col3:
#             pass

# def add_modify_delete_contacts():
#     st.title("Add/Modify/Delete Contact Details")
#     if 'name' not in st.session_state:
#         st.session_state['name'] = ""
#     if 'number' not in st.session_state:
#         st.session_state['number'] = ""

#     st.text_input("Enter Name", st.session_state['name'], key='input_name')
#     st.text_input("Enter Number", st.session_state['number'], key='input_number')

#     if st.button("Add Contact"):
#         name = st.session_state['input_name']
#         number = st.session_state['input_number']
#         if name and number:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             metadata.append({
#                 'name': name,
#                 'number': number,
#                 'file': "",  # No file for direct contact addition
#                 'timestamp': timestamp
#             })
#             save_metadata(metadata)
#             st.success(f"Contact {name} added.")
#         else:
#             st.error("Please enter both name and number.")

# def display_contact_report():
#     st.title("Contact Details Report")
#     st.write("### All Contacts")
#     for i, entry in enumerate(metadata):
#         st.write(f"Name: {entry['name']}, Number: {entry['number']}, Timestamp: {entry.get('timestamp', 'N/A')}")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Delete", key=f"delete_report_{entry['name']}_{i}"):
#                 metadata.remove(entry)
#                 save_metadata(metadata)
#                 st.experimental_rerun()
#         with col2:
#             pass

# def main():
#     st.sidebar.title("Navigation")
#     menu = st.sidebar.radio("Go to", ["Home", "Add/Modify/Delete Contacts", "Contact Report", "Saved Images"])

#     if menu == "Home":
#         st.title("Face Detection and Capture")

#         ctx = webrtc_streamer(
#             key="example",
#             mode=WebRtcMode.SENDRECV,
#             rtc_configuration=RTC_CONFIGURATION,
#             video_processor_factory=VideoProcessor,
#             media_stream_constraints={"video": True, "audio": False},
#         )

#         if ctx.video_processor:
#             if st.button("Capture Image"):
#                 ctx.video_processor.capture_image = True

#             captured_frame = ctx.video_processor.get_captured_frame()
#             if captured_frame is not None:
#                 st.image(captured_frame, caption="Captured Image")

#                 st.session_state['name'] = st.text_input("Enter Name", st.session_state['name'])
#                 st.session_state['number'] = st.text_input("Enter Number", st.session_state['number'])

#                 if st.button("Save Image"):
#                     filename = save_frame(captured_frame, st.session_state['name'], st.session_state['number'])
#                     if filename:
#                         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                         metadata.append({
#                             'name': st.session_state['name'],
#                             'number': st.session_state['number'],
#                             'file': filename,
#                             'timestamp': timestamp
#                         })
#                         save_metadata(metadata)
#                         st.success(f"Image saved as {filename}")
#                     else:
#                         st.error("Failed to save image")

#     elif menu == "Add/Modify/Delete Contacts":
#         add_modify_delete_contacts()

#     elif menu == "Contact Report":
#         display_contact_report()

#     elif menu == "Saved Images":
#         display_saved_images()

# if __name__ == "__main__":
#     main()






import logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import os
from datetime import datetime
import json

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

def compute_face_histogram(image, face_region):
    x, y, w, h = face_region
    face = image[y:y+h, x:x+w]
    face_hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    face_hist = cv2.normalize(face_hist, face_hist).flatten()
    return face_hist

saved_face_histograms = []
for entry in metadata:
    img = cv2.imread(entry['file'])
    if img is None:
        continue
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
        try:
            frm = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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

                if best_match_score < 0.6:
                    cv2.putText(frm, f"{name} - {number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 0, 255), 2)

            if self.capture_image:
                self.capture_image = False
                self.captured_frame = frm

            return av.VideoFrame.from_ndarray(frm, format="bgr24")
        except Exception as e:
            logging.error(f"Error in recv: {e}")
            return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")

    def get_captured_frame(self):
        return self.captured_frame

def save_frame(frame, name, number):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_images/{name}_{number}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        logging.error(f"Error saving frame: {e}")
        return None

def display_saved_images():
    st.title("Saved Images")
    for entry in metadata:
        if entry['file']:
            st.image(entry['file'], caption=f"{entry['name']} - {entry['number']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Delete", key=f"delete_{entry['file']}"):
                if entry['file'] and os.path.exists(entry['file']):
                    os.remove(entry['file'])
                metadata.remove(entry)
                save_metadata(metadata)
                st.experimental_rerun()
        with col2:
            if st.button("Edit", key=f"edit_{entry['file']}"):
                st.session_state['editing'] = entry

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

def add_modify_delete_contacts():
    st.title("Add/Modify/Delete Contact Details")
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
    if 'number' not in st.session_state:
        st.session_state['number'] = ""
    if 'timestamp_date' not in st.session_state:
        st.session_state['timestamp_date'] = ""
    if 'timestamp_time' not in st.session_state:
        st.session_state['timestamp_time'] = ""

    st.text_input("Enter Name", st.session_state['name'], key='input_name')
    st.text_input("Enter Number", st.session_state['number'], key='input_number')
    st.date_input("Enter Date", key='input_date')
    st.time_input("Enter Time", key='input_time')

    if st.button("Add Contact"):
        name = st.session_state['input_name']
        number = st.session_state['input_number']
        date = st.session_state['input_date']
        time = st.session_state['input_time']
        if name and number and date and time:
            timestamp = f"{date} {time}"
            metadata.append({
                'name': name,
                'number': number,
                'file': "",  # No file for direct contact addition
                'timestamp': timestamp
            })
            save_metadata(metadata)
            st.success(f"Contact {name} added.")
        else:
            st.error("Please enter name, number, date, and time.")

def display_contact_report():
    st.title("Contact Details Report")
    st.write("### All Contacts")
    for i, entry in enumerate(metadata):
        st.write(f"Name: {entry['name']}, Number: {entry['number']}, Timestamp: {entry.get('timestamp', 'N/A')}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete", key=f"delete_report_{entry['name']}_{i}"):
                metadata.remove(entry)
                save_metadata(metadata)
                st.experimental_rerun()
        with col2:
            pass

def main():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Home", "Add/Modify/Delete Contacts", "Contact Report", "Saved Images"])

    if menu == "Home":
        st.title("Face Detection and Capture")

        ctx = webrtc_streamer(
            key="example",
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
                st.image(captured_frame, caption="Captured Image")

                st.session_state['name'] = st.text_input("Enter Name", st.session_state['name'])
                st.session_state['number'] = st.text_input("Enter Number", st.session_state['number'])

                if st.button("Save Image"):
                    filename = save_frame(captured_frame, st.session_state['name'], st.session_state['number'])
                    if filename:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        metadata.append({
                            'name': st.session_state['name'],
                            'number': st.session_state['number'],
                            'file': filename,
                            'timestamp': timestamp
                        })
                        save_metadata(metadata)
                        st.success(f"Image saved as {filename}")
                    else:
                        st.error("Failed to save image")

    elif menu == "Add/Modify/Delete Contacts":
        add_modify_delete_contacts()

    elif menu == "Contact Report":
        display_contact_report()

    elif menu == "Saved Images":
        display_saved_images()

if __name__ == "__main__":
    main()
