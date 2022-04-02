import cv2
import numpy as np
import random

img = cv2.imread('coins.jpg')
blur = cv2.GaussianBlur(img, (5,5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
kernel = np.ones((5,5))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

background = cv2.dilate(closing, None, iterations=3)
# cv2.imshow('background', background)

distMap = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
cv2.normalize(distMap, distMap, 0.0, 255.0, cv2.NORM_MINMAX)
distMap = np.uint8(distMap)
# cv2.imshow("distMap", distMap)

foreground = cv2.threshold(distMap, 100, 255, cv2.THRESH_BINARY)[1]
foreground = cv2.erode(foreground, None, 2)
# cv2.imshow("foreground", foreground)

unknowZones = background-foreground
# cv2.imshow('unknow', unknowZones)

# danh label cac vung nen thi danh 0, cac vung khac bat dau tu 1
ret, markers = cv2.connectedComponents(foreground)
# label backgound=0, watershed coi la vung unknow, nen danh so nguyen khac
markers = markers + 1
# label vi tri unknow(cac vung giao nhau chua xac dinh) = 0
markers[unknowZones == 255] = 0
markers = cv2.watershed(img, markers)
# vung giao nhau label = -1
img[markers==-1] = (0,0,255)

# label = 11 bao gom ca vung giao nhau label=-1, background=1
label = []
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        if markers[i,j] not in label:
            label.append(markers[i,j])

# print(len(label))
colors = []
for contour in range(len(label)):
    colors.append((random.randint(0,256), random.randint(0,256), random.randint(0,256)))

dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index < len(label):
            dst[i,j,:] = colors[index-1]
cv2.imshow('result', dst)
cv2.putText(img, 'Count: {}'.format(len(label)-2), (20,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)

# cv2.imshow('open', closing)
# cv2.imshow('thresh', thresh)

cv2.imshow('image', img)
cv2.waitKey(0)