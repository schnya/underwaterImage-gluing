import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

DIRNAME: str = "2019_10_01 Aw 1/"


def drawMatches(imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
    height, qWidth = imgQuery.shape[:2]
    tHeight, tWidth = imgTrain.shape[:2]
    assert height == tHeight

    imageArray = np.zeros((height, qWidth + tWidth, 3), np.uint8)
    imageArray[:height, :qWidth] = imgQuery
    imageArray[:height, qWidth:] = imgTrain

    for idx, m in enumerate(matches):
        # 座標がぱっと見合うまで1個だけで試してる
        # if idx > 0:
        #     break

        px, py = queryKeyPoints[m[0].queryIdx].pt
        nx, ny = trainKeyPoints[m[0].trainIdx].pt
        px, py = int(px), int(py)
        nx, ny = int(nx), int(ny)

        cv2.drawMarker(
            imageArray,
            (px, py),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=5,
            line_type=cv2.LINE_8,
        )

        cv2.drawMarker(
            imageArray,
            (qWidth + nx, ny),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=5,
            line_type=cv2.LINE_8,
        )

        # Draw a small circle at both co-ordinates
        # colour blue
        # radius 4
        # thickness = 1
        cv2.circle(imageArray, (px, py), 4, (255, 255, 0), 1)
        cv2.circle(imageArray, (qWidth + nx, ny), 4, (255, 255, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(imageArray, (px, py), (qWidth + nx, ny), (0, 255, 0), 1)

    cv2.imwrite(f"{px, py}-{nx, ny}.png", imageArray)


def collage(imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
    height, qWidth = imgQuery.shape[:2]
    tHeight, tWidth = imgTrain.shape[:2]
    assert height == tHeight

    imageArray = np.zeros((height, qWidth + tWidth, 3), np.uint8)
    imageArray[:height, :qWidth] = imgQuery
    imageArray[:height, qWidth:] = imgTrain

    for idx, m in enumerate(matches):
        # 座標がぱっと見合うまで1個だけで試してる
        # if idx > 0:
        #     break

        prev = queryKeyPoints[m[0].queryIdx]
        next = trainKeyPoints[m[0].trainIdx]
        px, py = {int(v) for v in prev.pt}
        nx, ny = {int(v) for v in next.pt}

        # cv2.drawMarker(
        #     imageArray,
        #     (px, py),
        #     (255, 0, 255),
        #     markerType=cv2.MARKER_CROSS,
        #     markerSize=40,
        #     thickness=5,
        #     line_type=cv2.LINE_8,
        # )

        # cv2.drawMarker(
        #     imageArray,
        #     (qWidth + nx, ny),
        #     (255, 0, 255),
        #     markerType=cv2.MARKER_CROSS,
        #     markerSize=40,
        #     thickness=5,
        #     line_type=cv2.LINE_8,
        # )

        # Draw a small circle at both co-ordinates
        # colour blue
        # radius 4
        # thickness = 1
        cv2.circle(imageArray, (px, py), 4, (255, 255, 0), 1)
        cv2.circle(imageArray, (qWidth + nx, ny), 4, (255, 255, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(imageArray, (px, py), (qWidth + nx, ny), (0, 255, 0), 1)

    cv2.imwrite(f"{prev.pt}-{next.pt}.png", imageArray)


def drawMarker(img, keyPoints, matches, char):
    match = matches[0][0]
    if char == "q":
        # kp = keyPoints[match.queryIdx]
        (x, y) = keyPoints[match.queryIdx].pt
    else:
        # kp = keyPoints[match.trainIdx]
        (x, y) = keyPoints[match.trainIdx].pt

    (pt_x, pt_y) = int(x), int(y)
    # pt_x, pt_y = {int(v) for v in kp.pt}
    cv2.drawMarker(
        img,
        (pt_x, pt_y),
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=5,
        line_type=cv2.LINE_8,
    )

    cv2.imwrite(f"{pt_x}-{pt_y}.png", img)


if __name__ == "__main__":
    # Loads an image from a file.
    img_query = cv2.imread(DIRNAME + "PA011551.jpg")  # query
    img_train = cv2.imread(DIRNAME + "PA011552.jpg")  # train
    sift = cv2.SIFT.create()
    qKeyPoints, qDescriptors = sift.detectAndCompute(
        cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB), None
    )
    t_kp, t_des = sift.detectAndCompute(
        cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB), None
    )
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(qDescriptors, t_des, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)

    drawMarker(img_query, qKeyPoints, matches, "q")
    drawMarker(img_train, t_kp, matches, "t")

    # drawMatches(img_query, qKeyPoints, img_train, tKeyPoints, matches)
    # collage(img_query, qKeyPoints, img_train, tKeyPoints, matches)
