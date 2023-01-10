import argparse
import cv2
import glob
import numpy as np
from datetime import datetime
from fetchMatches import fetchMatches

from imageCompression import compressInputImg

DIRNAME: str = "./2019_10_01Aw1"
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="ディレクトリ名")
parser.add_argument("-n", "--count", default=10, type=int, help="使う画像の枚数")
parser.add_argument("-s", "--skip",  default=0,  type=int, help="スキップする画像の枚数")


class App:
    dir_name: str
    now: str
    n: int
    s: int

    def __init__(self, args: argparse.Namespace) -> None:
        self.now = datetime.now().strftime("%m%d_%H時%M")
        self.dir_name = args.dir
        self.n = args.count
        self.s = args.skip

    def stitch(self, imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
        base_h, base_w, _ = imgQuery.shape
        img_h, img_w, _ = imgTrain.shape

        x, y, K = 0, 0, 5
        for m in matches[:K]:
            m = m[0]

            px, py = queryKeyPoints[m.queryIdx].pt
            px, py = int(px), int(py)

            nx, ny = trainKeyPoints[m.trainIdx].pt
            nx, ny = int(nx), int(ny)

            x += px - nx
            y += py - ny
        x, y = int(x / K), int(y / K)

        height, width = max(base_h, abs(y) + img_h), max(base_w, abs(x) + img_w)
        imageArray = np.zeros((height, width, 3), np.uint8)
        print(f"{y} + {img_h}, {x} + {img_w} = {imageArray.shape}")

        if x >= 0 and y >= 0:
            imageArray[:height, :width] = imgQuery
            imageArray[y: y + img_h, x: x + img_w] = imgTrain
        elif x >= 0 and y < 0:
            imageArray[abs(y): abs(y) + base_h, :width] = imgQuery   
            imageArray[:img_h, x: x + img_w] = imgTrain
        elif x < 0 and y >= 0:
            imageArray[:height, abs(x): abs(x) + img_w] = imgQuery
            imageArray[y: y + img_h, :img_w] = imgTrain
        elif x < 0 and y < 0:
            imageArray[abs(y): abs(y) + img_h, abs(x): abs(x) + img_w] = imgQuery
            imageArray[:img_h, :img_w] = imgTrain
        
        return imageArray

    def save(self, img) -> None:
        cv2.imwrite(
            f"output/{self.dir_name}|{self.now}|n={self.n}|s={self.s}.jpg", img)


if __name__ == "__main__":
    args = parser.parse_args()
    assert (args.dir)

    imgPaths = sorted(
        glob.glob(f"./{args.dir}/*.JPG"))[args.skip:args.skip+args.count]
    output = compressInputImg(imgPaths[0])

    app = App(args)
    for path in imgPaths[1:]:
        t_img = compressInputImg(path)
        q_key_point, t_key_point, matches = fetchMatches(output, t_img)
        output = app.stitch(output, q_key_point, t_img, t_key_point, matches)

    app.save(output)
    exit()
