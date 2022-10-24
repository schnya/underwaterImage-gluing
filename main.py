import cv2
import numpy as np

DIRNAME: str = "2019_10_01 Aw 1/"
sift = cv2.SIFT.create()
bf = cv2.BFMatcher()
bgr2rgb = cv2.COLOR_BGR2RGB


def findMatches(imgQuery, imgTrain):
    q_kp, q_des = sift.detectAndCompute(cv2.cvtColor(imgQuery, bgr2rgb), None)
    t_kp, t_des = sift.detectAndCompute(cv2.cvtColor(imgTrain, bgr2rgb), None)
    matches = bf.knnMatch(q_des, t_des, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)

    return matches, q_kp, t_kp


def collage(imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
    height, qWidth = imgQuery.shape[:2]
    tHeight, tWidth = imgTrain.shape[:2]

    px, py = queryKeyPoints[matches[0][0].queryIdx].pt
    nx, ny = trainKeyPoints[matches[0][0].trainIdx].pt
    px, py = int(px), int(py)
    nx, ny = int(nx), int(ny)

    iHeight, iWidth = py + tHeight - ny, px + tWidth - nx
    imageArray = np.zeros((height + iHeight, qWidth + iWidth, 3), np.uint8)
    imageArray[:height, :qWidth] = imgQuery
    imageArray[py:iHeight, px:iWidth] = imgTrain[ny:, nx:]
    cv2.drawMarker(
        imageArray,
        (px, py),
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=5,
        line_type=cv2.LINE_8,
    )
    cv2.line(imageArray, (px, py), (px, iHeight), (0, 255, 0), 1)

    return imageArray


if __name__ == "__main__":
    # Loads an image from a file.
    img1 = cv2.imread(DIRNAME + "comp_PA011555.jpg")
    img2 = cv2.imread(DIRNAME + "comp_PA011554.jpg")
    img3 = cv2.imread(DIRNAME + "comp_PA011553.jpg")
    img4 = cv2.imread(DIRNAME + "comp_PA011552.jpg")
    img5 = cv2.imread(DIRNAME + "comp_PA011551.jpg")

    ### Finding matches between the 2 images and their keypoints
    matches, qKeyPoints, tKeyPoints = findMatches(img1, img2)
    img_new = collage(img1, qKeyPoints, img2, tKeyPoints, matches)

    matches, qKeyPoints, tKeyPoints = findMatches(img_new, img3)
    img_new = collage(img_new, qKeyPoints, img3, tKeyPoints, matches)

    matches, qKeyPoints, tKeyPoints = findMatches(img_new, img4)
    img_new = collage(img_new, qKeyPoints, img4, tKeyPoints, matches)

    matches, qKeyPoints, tKeyPoints = findMatches(img_new, img5)
    img_new = collage(img_new, qKeyPoints, img5, tKeyPoints, matches)

    cv2.imwrite("output.png", img_new)
