import cv2
import glob
import numpy as np
from datetime import datetime

from imageCompression import imgEncodeDecode

DIRNAME: str = "./2562_06_12RachaYai"
sift = cv2.SIFT.create()
bf = cv2.BFMatcher()
bgr2rgb = cv2.COLOR_BGR2RGB


def collage(imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
    x, y = 0, 0
    for m in matches[:3]:
        m = m[0]

        px, py = queryKeyPoints[m.queryIdx].pt
        px, py = int(px), int(py)

        nx, ny = trainKeyPoints[m.trainIdx].pt
        nx, ny = int(nx), int(ny)

        # TODO: 逆の時死ぬ
        x += px - nx
        y += py - ny
    x, y = int(x / 3), int(y / 3)

    base_h, base_w, _ = imgQuery.shape
    img_h, img_w, _ = imgTrain.shape

    # if base_h > y + img_h:
    #     return imgQuery
    height, width = y + img_h, max(base_w, x + img_w)
    imageArray = np.zeros((height, width, 3), np.uint8)

    print(f"{y} + {img_h}, {x} + {img_w} = {imageArray.shape}")
    imageArray[:base_h, :base_w] = imgQuery
    imageArray[y : img_h + y, x : img_w + x] = imgTrain

    # cv2.line(imageArray, (x, y), (x + img_w, y + img_h), (0, 255, 0), 2)

    return imageArray


def fetchMatches(imgQuery, file_path):
    imgTrain = imgEncodeDecode(file_path)
    q_kp, q_des = sift.detectAndCompute(cv2.cvtColor(imgQuery, bgr2rgb), None)
    t_kp, t_des = sift.detectAndCompute(cv2.cvtColor(imgTrain, bgr2rgb), None)
    matches = bf.knnMatch(q_des, t_des, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance)

    return imgTrain, matches, q_kp, t_kp


if __name__ == "__main__":
    n = 10
    filenames = sorted(glob.glob(f"{DIRNAME}/*.JPG"), reverse=True)[:n]

    output = imgEncodeDecode(filenames[0])
    for name in filenames[1:]:
        train_img, matches, qKeyPoints, tKeyPoints = fetchMatches(output, name)
        output = collage(output, qKeyPoints, train_img, tKeyPoints, matches)

    today = datetime.now()
    today = today.isoformat().split("T")[0]
    cv2.imwrite(f"{today}|n={n}|{DIRNAME}.jpg", output)
