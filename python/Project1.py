import cv2
from cv2 import findContours
import numpy as np

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)     # id number 3 : width
cap.set(4, frameHeight)     # id number 4 : height

myColors = [[163, 110, 0, 179, 255, 255],   # RED
            [93, 61, 0, 126, 255, 255],     # BLUE
            # [11, 114, 171, 28, 255, 255]]   # YELLOW
            [0, 107, 0, 19, 255, 255]]      # ORANGE

# BGR 형태로 저장
myColorValues = [[0, 0, 255],       # RED
                 [255, 0, 0],       # BLUE
                 [0, 128, 255]]     # ORANGE

def findColor(img, myColors, myColorValues) :
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors :
        lower = np.array(color[:3])
        upper = np.array(color[3:])
        mask = cv2.inRange(imgHSV, lower, upper)
        # mask를 getContours() 함수에 매개변수로 넣어줌
        x, y = getContours(mask)
        # myColorValues의 인덱스(count)가 증가함에 따라, 원의 색깔이 해당 색상으로 변하게 됨
        cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0 :
            newPoints.append([x, y, count])
        count += 1

    return newPoints

# Chap8의 getContours() 함수
def getContours(img) :
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # area > 500이 아닌 경우에도 값이 출력되어야 하므로 아래와 같이 지정해줌
    x, y, w, h = 0, 0, 0, 0

    for cnt in contours :
        area = cv2.contourArea(cnt)

        if area > 500 :
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)

    # findColor() 함수의 cv2.circle() 함수에 사용될 포인트 return
    return x+w//2, y

def drawOnCanvas(myPoints, myColorValues) :
    for point in myPoints :
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)
        

myPoints = []   # [x, y, colorId]

while True :
    success, img = cap.read()
    # 원본 이미지를 imgResult
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0 :
        for newP in newPoints :
            myPoints.append(newP)
    if len(myPoints) != 0 :
        drawOnCanvas(myPoints, myColorValues)


    cv2.imshow("Result", imgResult)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break