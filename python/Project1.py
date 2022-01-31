# Webcam 사용
import cv2

frameWidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)     # id number 3 : width
cap.set(4, frameheight)     # id number 4 : height
cap.set(10, 150)    # id number 10 : brightness

while True :
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break