import cv2
import numpy as np

img = cv2.imread("img/img1/lena.jpg")

###########################################
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
###########################################


imgStack1 = stackImages(1, ([img, img, img]))
imgStack2 = stackImages(1, ([img, img, img], [img, img, img]))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = stackImages(1, ([img, imgGray, img], [img, img, img]))


# imgHor = np.hstack((img, img, img))
# imgVer = np.vstack((img, img))

# # 두개의 이미지가 수평하게 출력됨
# cv2.imshow("Horizontal", imgHor)
# # 두개의 이미지가 수직으로 출력됨
# cv2.imshow("Vertical", imgVer)

cv2.imshow("Image Stack1", imgStack1)
cv2.imshow("Image Stack2", imgStack2)
cv2.imshow("Image Gray", img_gray)

cv2.waitKey(0)