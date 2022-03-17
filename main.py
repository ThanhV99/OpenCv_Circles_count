import cv2
import numpy as np

img = cv2.imread('coins.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.medianBlur(img_gray, 5)
canny = cv2.Canny(img_gray, 200, 230)
circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=20, minRadius=1, maxRadius=50)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(img, (x,y), 1, (0,0,255), 3)
        cv2.circle(img, (x,y), r, (0,255,0), 3)

assert circles.shape is not None, "circles None"
so_luong = circles.shape[1]
cv2.putText(img, f'Count:{so_luong}', (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

cv2.imshow('canny', canny)
cv2.imshow('img', img)

cv2.waitKey(0)
