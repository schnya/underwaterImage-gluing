from cmath import pi
import cv2
import numpy as np
import matplotlib.pyplot as plt


def findMatches(queryImage, trainImage):
    # Using SIFT to find the keypoints and decriptors in the images
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    queryImage_kp, queryImage_des = sift.detectAndCompute(
        cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY), None
    )
    trainImage_kp, trainImage_des = sift.detectAndCompute(
        cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY), None
    )

    # Brute-Force matcher with default params
    bf = cv2.BFMatcher()
    initialMatches = bf.knnMatch(queryImage_des, trainImage_des, k=2)
    print("initialMatches.length:", initialMatches.__len__())

    # Apply ratio test and filter out the good matches.
    goodMatches = []
    for m, n in initialMatches:
        """
        type(m), type(n)は <class 'cv2.DMatch'>
        DMatchが持つプロパティは以下の4つ
        ・distance: 特徴量記述子間の距離．低いほど良い
        ・trainIdx: 学習記述子（参照データ）の記述子のインデックス
        ・queryIdx: クエリ記述子（検索データ）の記述子のインデックス
        ・imgIdx:   学習画像のインデックス
        """
        if m.distance < 0.75 * n.distance:
            goodMatches.append([m])

    return goodMatches, queryImage_kp, trainImage_kp


if __name__ == "__main__":
    print(cv2.__version__)
    dirname = "OneDrive_1_6-28-2022/"

    # queryImage and trainImage
    img1 = cv2.imread(dirname + "comp_PA011324.jpg")  # query
    img2 = cv2.imread(dirname + "comp_PA011325.jpg")  # train

    # Finding matches between the 2 images and their keypoints
    matches, img1_kp, img2_kp = findMatches(img1, img2)
    # If less than 4 matches found, exit the code.
    if len(matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    print(img1_kp.__len__())
    img1_kp = sorted(img1_kp, key=lambda f: f.angle)[:75]
    img2_kp = sorted(img2_kp, key=lambda f: f.angle)[:75]

    for index, (one, two) in enumerate(zip(img1_kp, img2_kp)):
        print(index)
        print("angle", one.angle, two.angle)
        print("pt", one.pt, two.pt)
        print("responose", one.response, two.response)
        print("size", one.size, two.size)
        print("- - - - - - - - - -")

    # cv2.drawMatchesKnn expects list of lists as matches.
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)

    img3 = cv2.drawMatchesKnn(
        img1, img1_kp, img2, img2_kp, matches, None, **draw_params
    )
    plt.imshow(img3)
    plt.show()
