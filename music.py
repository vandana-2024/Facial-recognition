import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.set_page_config(
    page_title="Emotion Based Music Recommender",
    page_icon="😊",
    layout="wide"
)

st.title("Emotion Based Music Recommender")

# State variables
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False

# Login page
if not st.session_state["is_logged_in"]:
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        st.session_state["is_logged_in"] = True
        st.success("You are now logged in.")

# Continue with the app if logged in
if st.session_state["is_logged_in"]:
    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion=""

    if not(emotion):
        st.session_state["run"] = "true"
    else:
        st.session_state["run"] = "false"

    class EmotionProcessor:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")

            # Process the frame for emotion detection
            frm = cv2.flip(frm, 1)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]

                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                np.save("emotion.npy", np.array([pred]))

            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

            return av.VideoFrame.from_ndarray(frm, format="bgr24")

    # Sidebar for inputs
    st.sidebar.title("User Input")
    lang = st.sidebar.text_input("Language")
    singer = st.sidebar.text_input("Singer")

    if lang and singer:
        webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

    btn = st.sidebar.button("Recommend me songs")

    if btn:
        if st.session_state["run"] == "true":
            st.warning("Please let me capture your emotion first")
        else:
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
            np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"
