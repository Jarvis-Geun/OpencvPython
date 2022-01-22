import cv2
import numpy as np

img = cv2.imread("img/img2/lambo.jpg")
print(img.shape)

imgResize = cv2.resize(img, (300, 200))     # 300 : width, 200 : height
print(imgResize.shape)

imgCropped = img[0:200, 200:500]        # 0:200 => height, 200:500 => width

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)