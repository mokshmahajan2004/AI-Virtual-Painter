import cv2
import numpy as np
import mediapipe as mp
import os
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Variables
brushThickness = 15
eraserThickness = 50
folderPath = 'Header'

# Load header images
myList = os.listdir(folderPath)
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        self.header = overLayList[0]
        self.drawColor = (255, 0, 255)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def process_frame(self, frame):
        img = cv2.flip(frame, 1)
        img = self.findHands(img)
        lmList = self.findPosition(img, draw=False)
        if lmList:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = self.fingersUp()
            if fingers[1] and fingers[2]:
                if y1 < 125:
                    if 250 < x1 < 450:
                        self.header = overLayList[0]
                        self.drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        self.header = overLayList[1]
                        self.drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        self.header = overLayList[2]
                        self.drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        self.header = overLayList[3]
                        self.drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, self.drawColor, cv2.FILLED)
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1
                if self.drawColor == (0, 0, 0):
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, eraserThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, eraserThickness)
                else:
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, brushThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, brushThickness)
                self.xp, self.yp = x1, y1

            imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, self.imgCanvas)

        img[0:125, 0:1280] = self.header
        return img

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.85)
    
    def transform(self, frame):
        img = self.detector.process_frame(frame)
        return img

def main():
    st.title('AI Virtual Painter')
    st.write('Use your hand to select color and draw.')
    st.write('Use two fingers to select color and one finger to draw.')

    st.subheader("Webcam Stream")

    # Create a Streamlit placeholder for the video
    video_placeholder = st.empty()

    # Stream the video using webrtc_streamer
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    # Video placeholder usage
    if webrtc_ctx.video_transformer:
        frame = webrtc_ctx.video_transformer.get_frame()
        if frame is not None:
            video_placeholder.image(frame, channels='BGR', use_column_width=True)

    # If a quit button is pressed, stop the app
    if st.button('Quit', key='quit_button'):
        st.stop()

if __name__ == "__main__":
    main()
