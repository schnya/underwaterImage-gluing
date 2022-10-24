import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

DIRNAME: str = "2019_10_01 Aw 1/"


def findMatches(imgQuery, imgTrain):
    # Create for extracting keypoints and computing descriptors
    # using the SIFT algorithm by David G. Lowe.
    # Read More: Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vision, 60(2):91–110, November 2004.
    sift = cv2.SIFT.create()

    # Detects keypoints and computes their descriptors
    qKeyPoints, qDescriptors = sift.detectAndCompute(
        cv2.cvtColor(imgQuery, cv2.COLOR_BGR2RGB), None
    )
    tKeyPoints, tDescriptors = sift.detectAndCompute(
        cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB), None
    )

    ### Brute-Force matcher with default params
    bf = cv2.BFMatcher()
    ## 何順？閾値が上位の順？
    initialMatches = bf.knnMatch(qDescriptors, tDescriptors, k=2)
    # print("initialMatches.length:", initialMatches.__len__()) # 274

    ### Apply ratio test and filter out the good matches.
    goodMatches = []
    imgHeight, imgWidth = imgQuery.shape[:2]
    for m, n in initialMatches:
        """
        type(m), type(n)は <class 'cv2.DMatch'>
        DMatchが持つプロパティは以下の4つ
        ・distance: 特徴量記述子間の距離．低いほど良い
        ・trainIdx: 学習記述子（参照データ）の記述子のインデックス
        ・queryIdx: クエリ記述子（検索データ）の記述子のインデックス
        ・imgIdx:   学習画像のインデックス 0
        """
        x, y = {int(v) for v in qKeyPoints[m.queryIdx].pt}
        if x < imgWidth / 2 and y < imgHeight and m.distance < 0.6 * n.distance:
            goodMatches.append([m])

    goodMatches = sorted(goodMatches, key=lambda x: x[0].distance)
    # print(goodMatches[0][0].distance)  # 71
    # print(goodMatches[-1][0].distance)  # 206
    print(goodMatches[0][0].trainIdx)
    return goodMatches, qKeyPoints, tKeyPoints


def drawMatchesKnn(imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
    # cv2.drawMatchesKnn expects list of lists as matches.
    params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    img = cv2.drawMatches(
        imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches, None, **params
    )
    cv2.imwrite("drawMatchesKnn.png", img)


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
    for idx, m in enumerate(matches):
        # 座標がぱっと見合うまで1個だけで試してる
        if idx > 0:
            break

        if char == "q":
            (x, y) = keyPoints[m[0].queryIdx].pt
        else:
            (x, y) = keyPoints[m[0].trainIdx].pt
        pt_x, pt_y = int(x), int(y)

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

    ### Finding matches between the 2 images and their keypoints
    matches, qKeyPoints, tKeyPoints = findMatches(img_query, img_train)

    ### If less than 4 matches found, exit the code.
    if len(matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)
    else:
        print(f"Key Points: {len(qKeyPoints)}個と{len(tKeyPoints)}個中、")
        print(f"マッチしたKey Pointsは {len(matches)}個")

    drawMarker(img_query, qKeyPoints, matches, "q")
    drawMarker(img_train, tKeyPoints, matches, "t")

    # drawMatchesKnn(img_query, qKeyPoints, img_train, tKeyPoints, matches)
    # collage(img_query, qKeyPoints, img_train, tKeyPoints, matches)