import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# from models import model
import tensorflow as tf
import pickle



with open('alphaCnn.pickle', 'rb') as fl:
    alpmodel = pickle.load(fl)

# variables

width, height = 1280, 720
folderPath = "Blank"

words = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

# camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

success, img = cap.read()


# list of images
pathImages = sorted(os.listdir(folderPath), key=len)

imgNumber = 0
ws, hs = int(166 * 1.2), int(100 * 1.2)
gestureThreshold = 400
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

IMG_SIZE = 28
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
text = "Some text in a box!"
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x+text_width+2, text_offset_y-text_height-2))
cntr = -1

while True:
    # import images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    cntr += 1

    # recognition

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 5)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        lmList = hand['lmList']

        indexFinger = lmList[8][0], lmList[8][1]

        if cy <= gestureThreshold:  # if hand is at height of face

            # gesture 1 - left
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber -= 1

            # gesture 2 - right
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        # gesture 3 - show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 0), cv2.FILLED)
            annotationStart = False

        # gesture 4 - draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 0), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # gesture 5 - erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber > -1:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart = False

    # button pressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 0), 12)

    if cntr%2 == 0:

        img2 = cv2.GaussianBlur(imgCurrent, (7,7), 0)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        newimg = cv2.resize(img_thresh, (28,28), interpolation=cv2.INTER_AREA)
        newimg = np.reshape(newimg, (1,28,28,1))



        prediction1 = alpmodel.predict(newimg)
        status = words[np.argmax(prediction1)]


    x1, y1, w1, h1 = 0, 0, 175, 75
    cv2.rectangle(imgCurrent, (x1, x1), (x1+w1, y1+h1), (0, 255, 0), -1)
    cv2.putText(imgCurrent, str(status), (x1+int(w1/5), y1+int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # adding webcam image on slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, chan = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
