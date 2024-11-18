import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit app title
st.title("Hand Gesture Drawing App")

# Sidebar for color selection and actions
st.sidebar.title("Controls")
selected_color = st.sidebar.radio(
    "Select Color", ["Blue", "Green", "Red", "Yellow"], index=0
)
clear_button = st.sidebar.button("Clear Canvas")

# Assign color index based on selection
color_map = {"Blue": 0, "Green": 1, "Red": 2, "Yellow": 3}
colorIndex = color_map[selected_color]

# Initialize deque arrays for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Color indexes for tracking
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Create a blank canvas
paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255

# Mediapipe initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Define video transformer for Streamlit WebRTC
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.bpoints = bpoints
        self.gpoints = gpoints
        self.rpoints = rpoints
        self.ypoints = ypoints
        self.blue_index = blue_index
        self.green_index = green_index
        self.red_index = red_index
        self.yellow_index = yellow_index
        self.paintWindow = paintWindow
        self.colorIndex = colorIndex

    def transform(self, frame):
        global paintWindow
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        framergb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * img.shape[1])
                    lmy = int(lm.y * img.shape[0])
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])
            thumb = (landmarks[4][0], landmarks[4][1])

            # Check for drawing action
            if thumb[1] - fore_finger[1] < 30:
                self.bpoints.append(deque(maxlen=1024))
                self.blue_index += 1
                self.gpoints.append(deque(maxlen=1024))
                self.green_index += 1
                self.rpoints.append(deque(maxlen=1024))
                self.red_index += 1
                self.ypoints.append(deque(maxlen=1024))
                self.yellow_index += 1
            else:
                if self.colorIndex == 0:
                    self.bpoints[self.blue_index].appendleft(fore_finger)
                elif self.colorIndex == 1:
                    self.gpoints[self.green_index].appendleft(fore_finger)
                elif self.colorIndex == 2:
                    self.rpoints[self.red_index].appendleft(fore_finger)
                elif self.colorIndex == 3:
                    self.ypoints[self.yellow_index].appendleft(fore_finger)

        points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(img, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(self.paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        return img


# Initialize the WebRTC streamer
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Show the paint window
st.image(paintWindow, caption="Paint Window", channels="BGR")
