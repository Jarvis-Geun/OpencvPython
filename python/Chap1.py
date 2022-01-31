# import cv2
# print("Package Imported")

# img = cv2.imread("img/img1/lena.jpg")
# cv2.imshow("Output", img)
# cv2.waitKey(0)

##########################################

## Video 출력
# import cv2

# cap = cv2.VideoCapture("./Resources/test_video.mp4")

# while True :
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q') :
#         break

##########################################

# Webcam 사용
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)     # id number 3 : width
cap.set(4, 480)     # id number 4 : height
cap.set(10, 100)    # id number 10 : brightness

while True :
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break