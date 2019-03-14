import numpy as np
import cv2

img = cv2.imread('./test.png')
print(img.shape)
# cv2.imshow("new",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
# cv2.imshow("new",gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("new",thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

markers = markers+1

markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
# cv2.imshow("new",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()