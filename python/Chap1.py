import cv2
print("Package Imported")

img = cv2.imread("img/img1/lena.jpg")
cv2.imshow("Output", img)
cv2.waitKey(0)