import numpy as np
import cv2 as cv

img = cv.imread("signature_1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite("sift_keypoints.jpg", img)


def visual(des, kp):
    print(des)
    print(kp)

visual(des, kp)
