import numpy as np
import cv2
import matplotlib.pyplot as plt



def segmentation(imag):
    blurred = cv2.GaussianBlur(imag, (5, 5), 0)
    histogram = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    hist_normalize = histogram.ravel() / histogram.max()
    Q = hist_normalize.cumsum()
    x_axis = np.arange(256)
    mini = np.inf
    epsilon = 10^(-8)
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_normalize, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        b1, b2 = np.hsplit(x_axis, [i])
        m1, m2 = np.sum(p1 * b1) / (q1+epsilon), np.sum(p2 * b2) / (q2+epsilon)
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / (q1+epsilon), np.sum(((b2 - m2) ** 2) * p2) / (epsilon+q2)
        fn = v1 * q1 + v2 * q2
        if fn < mini:
            mini = fn
    ret, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized

# imag = cv2.imread('./house.jpg',0)
# print(imag.shape)


vidcap = cv2.VideoCapture('output.avi')
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ss = segmentation(gray)
    plt.imshow(ss,"gray")
    plt.pause(0.1)
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
    count += 1
