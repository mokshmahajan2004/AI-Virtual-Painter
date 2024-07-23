import cv2
import numpy as np
import mediapipe as mp
import os
import streamlit as st
from PIL import Image

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

def process_frame(frame, header, drawColor, detector, brushThickness, eraserThickness):
    global xp, yp, imgCanvas

    img = cv2.flip(frame, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overLayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overLayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overLayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    return img, header, drawColor

def main():
    global xp, yp, imgCanvas

    st.title('AI Virtual Painter')
    st.write('Use your hand to select color and draw.')
    st.write('Use two fingers to select color and one finger to draw.')

    detector = HandDetector(detectionCon=0.85)
    drawColor = (255, 0, 255)
    header = overLayList[0]
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    xp, yp = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    video_placeholder = st.empty()
    quit_button_pressed = st.button('Quit', key='quit_button')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img, header, drawColor = process_frame(frame, header, drawColor, detector, brushThickness, eraserThickness)

        # Convert image to RGB and update the video placeholder
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_placeholder.image(img_rgb, channels='RGB', use_column_width=True)

        if quit_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
