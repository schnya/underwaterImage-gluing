import cv2

DIRNAME: str = "2019_10_01 Aw 1/"

img1 = cv2.imread(DIRNAME + "PA011551.jpg")
img2 = cv2.imread(DIRNAME + "PA011552.jpg")
shift = cv2.SIFT_create()
kp1, des1 = shift.detectAndCompute(img1, None)
kp2, des2 = shift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
matches = sorted(matches, key=lambda x: x[0].distance)
(x, y) = kp1[matches[0][0].queryIdx].pt
cv2.circle(img1, (int(x), int(y)), 30, (255, 0, 0), 3, 5)
cv2.imwrite("fig1.jpg", img1)
(x, y) = kp2[matches[0][0].trainIdx].pt
cv2.circle(img2, (int(x), int(y)), 30, (255, 0, 0), 3, 5)
cv2.imwrite("fig2.jpg", img2)
