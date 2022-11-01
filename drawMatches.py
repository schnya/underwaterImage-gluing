import numpy as np
import cv2
import glob

from main import fetchMatches

DIRNAME: str = "./2019_10_01Aw1"

PARAMS = dict(
    markerType=cv2.MARKER_CROSS,
    markerSize=40,
    thickness=5,
    line_type=cv2.LINE_8,
)


def drawMatches(current_img, queryKeyPoints, new_img, trainKeyPoints, matches):
    K = 10
    height, width, _ = current_img.shape
    imageArray = np.zeros((height + 1000, width + 1000 + 4000, 3), np.uint8)
    imageArray[:height, :width] = current_img
    imageArray[height - 2000 : height + 1000, width + 1000 :] = new_img

    for m in matches[:K]:
        m = m[0]

        px, py = queryKeyPoints[m.queryIdx].pt
        px, py = int(px), int(py)

        nx, ny = trainKeyPoints[m.trainIdx].pt
        nx, ny = width + 1000 + int(nx), height - 2000 + int(ny)

        cv2.drawMarker(imageArray, (px, py), (255, 0, 255), **PARAMS)
        cv2.drawMarker(imageArray, (nx, ny), (255, 0, 255), **PARAMS)
        cv2.circle(imageArray, (px, py), 4, (255, 255, 0), 2)
        cv2.circle(imageArray, (nx, ny), 4, (255, 255, 0), 2)
        cv2.line(imageArray, (px, py), (nx, ny), (0, 255, 0), 2)

    cv2.imwrite(f"drawMaches_{K}.jpg", imageArray)


if __name__ == "__main__":
    n = 10
    filename = sorted(glob.glob(f"{DIRNAME}/*.JPG"), reverse=True)[n]

    output = cv2.imread(f"n={n}.jpg")
    train_img, matches, qKeyPoints, tKeyPoints = fetchMatches(output, filename)
    drawMatches(output, qKeyPoints, train_img, tKeyPoints, matches)
