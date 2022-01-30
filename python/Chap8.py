import cv2
from cv2 import COLOR_BGR2GRAY
from matplotlib.pyplot import contourf
import numpy as np

#########################################
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
#########################################
def getContours(img) :
    # cv2.findContours(image, mode, method, ...)
    # RETR_EXTERNAL : contours line중 가장 바깥쪽 line만 찾음
    # CHAIN_APPROX_NONE : 모든 contours point를 저장
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("len(contours) :", len(contours))
    for cnt in contours :
        area = cv2.contourArea(cnt)
        print("area : ", area)

        if area > 500 : # area가 500 보다 크면 경계선을 그림
            # 파란색(255, 0, 0) 선으로 그림
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)

            # perimeter : 경계선의 호의 길이
            # cv2.arcLength(curve, closed)
            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print("len(approx) : ", len(approx))
            objCor = len(approx)

            # Bounding box 그리기
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3 : objectType = "Tri"

            elif objCor == 4 :
                aspRatio = w/float(h)
                # width와 height를 나누었을 때 대략 0.95 ~ 1.05의 값을 가지면 Square
                # 아니라면, Rectangle이라고 생각할 수 있음
                if aspRatio > 0.95 and aspRatio < 1.05 :
                    objectType = "Square"
                else :
                    objectType = "Rectangle"
            elif objCor == 5 :  objectType = "Pentagon"

            elif objCor > 5 : objectType = "Circle"

            else :  objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x+(w//2) - 25, y+(h//2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)


#########################################

path = 'img/img4/shapes.png'
img = cv2.imread(path)
# 원본 이미지까지 변화시키므로 copy해주어 다른 이미지에 저장함
imgContour = img.copy()

imgGray = cv2.cvtColor(img, COLOR_BGR2GRAY)
# GaussianBlur(src, ksize(=kernel size), sigmaX, ...)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# Canny() : Canny edge detection 사용
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)


# zeros_like() : Return an array of zeros with the same shape and type as a given array.
imgBlank = np.zeros_like(img)
imgStack = stackImages(0.6, ([img, imgGray, imgBlur],
                            [imgCanny, imgContour, imgBlank]))

cv2.imshow("Stack", imgStack)
cv2.waitKey(0)