from datetime import datetime
import cv2
import numpy as np

PARAMS = dict(
    markerType=cv2.MARKER_CROSS,
    markerSize=40,
    thickness=5,
    line_type=cv2.LINE_8,
)


def drawMatches(img_q, q_kp, img_t, t_kp, matches, dir_name):
    now = datetime.now().strftime("%m%d_%H時%M")
    height, width, _ = img_q.shape
    imageArray = np.zeros((height + 1000, width + 1000 + 4000, 3), np.uint8)
    imageArray[:height, :width] = img_q
    imageArray[height - 2000 : height + 1000, width + 1000 :] = img_t

    for m in matches:
        m = m[0]

        px, py = q_kp[m.queryIdx].pt
        px, py = int(px), int(py)

        nx, ny = t_kp[m.trainIdx].pt
        nx, ny = width + 1000 + int(nx), height - 2000 + int(ny)

        cv2.drawMarker(imageArray, (px, py), (255, 0, 255), **PARAMS)
        cv2.drawMarker(imageArray, (nx, ny), (255, 0, 255), **PARAMS)
        cv2.circle(imageArray, (px, py), 4, (255, 255, 0), 2)
        cv2.circle(imageArray, (nx, ny), 4, (255, 255, 0), 2)
        cv2.line(imageArray, (px, py), (nx, ny), (0, 255, 0), 2)

    cv2.imwrite(
        f"output/M{now}atches|{dir_name}.jpg",
        imageArray,
    )
