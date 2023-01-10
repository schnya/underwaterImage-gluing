from datetime import datetime
import cv2
from imageCompression import compressInputImg

sift = cv2.SIFT.create()
bf = cv2.BFMatcher()
bgr2rgb = cv2.COLOR_BGR2RGB

def fetchMatches(imgQuery, imgTrain, above: int = 50):
    q_kp, q_des = sift.detectAndCompute(cv2.cvtColor(imgQuery, bgr2rgb), None)
    t_kp, t_des = sift.detectAndCompute(cv2.cvtColor(imgTrain, bgr2rgb), None)

    print("Start knnMatch:", datetime.now())
    matches = bf.knnMatch(q_des, t_des, k=2)
    print("⏰End knnMatch:", datetime.now())

    matches = sorted(matches, key=lambda x: x[0].distance)[:above]

    return q_kp, t_kp, matches

def applyRatioTest(matches):
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good
