import cv2
import numpy as np

img = cv2.imread("img/img2/cards.jpg")

width, height = 250, 350
'''
그림판에서 커서를 카드 이미지의 꼭짓점에 위치시키면 왼쪽 하단에 픽셀좌표값이 표시됨.
이를 이용해서 꼭짓점 좌표를 확인한 후 아래와 같이 입력하면 됨.
'''
pts1 = np.float32([[355, 153], [489, 122], [455, 290], [596, 247]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))


cv2.imshow("Image", img)
cv2.imshow("Output", imgOutput)

cv2.waitKey(0)