import argparse
import cv2
import glob
from datetime import datetime
from driver.angleModeDriver import AngleModeDriver
from driver.meanDriver import MeanDriver
from drawMatches import drawMatches
from fetchMatches import fetchMatches
from port.port import Port

from imageCompression import compressInputImg

DIRNAME: str = "./2019_10_01Aw1"
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="ディレクトリ名")
parser.add_argument("-n", "--count", default=10, type=int, help="使う画像の枚数")
parser.add_argument("-s", "--skip",  default=0,  type=int, help="スキップする画像の枚数")
parser.add_argument("-a", "--angle", action="store_true")


class App:
    dir_name: str
    now: str
    n: int
    s: int
    calculate_port: Port

    def __init__(self, args: argparse.Namespace, port=MeanDriver()) -> None:
        self.now = datetime.now().strftime("%m%d_%H時%M")
        self.dir_name = args.dir
        self.n = args.count + 1
        self.s = args.skip
        self.calculate_port = port

    def generateListOfImgPath(self) -> list[str]:
        return sorted(glob.glob(f"./{self.dir_name}/*.JPG"))[self.s:self.s + self.n]

    def mosaic(self, imgQuery, queryKeyPoints, imgTrain, trainKeyPoints, matches):
        # distanceデカい奴はスキップしてみたけどあんま意味なさげ
        # print('最もmatchのdistance', matches[0][0].distance)
        # if matches[0][0].distance > 100:
        #     return imgQuery

        qY, qX, _ = imgQuery.shape
        tY, tX, _ = imgTrain.shape
        # print('Query Size', qY, qX, 'Train size', tY, tX)

        x, y = self.calculate_port.measureDistanceTraveledByKeyPoint(
            matches, queryKeyPoints, trainKeyPoints)

        new_img = self.calculate_port.createEmptyImage(qY, qX, tY, tX, x, y)

        if x >= 0 and y >= 0:
            new_img[:qY, :qX] = imgQuery
            new_img[y: y + tY, x: x + tX] = imgTrain
        elif x >= 0 and y < 0:
            new_img[abs(y):, :qX] = imgQuery
            new_img[:tY, x: x + tX] = imgTrain
        elif x < 0 and y >= 0:
            new_img[:qY, abs(x):] = imgQuery
            new_img[y: y + tY, :tX] = imgTrain
        elif x < 0 and y < 0:
            new_img[abs(y):, abs(x):] = imgQuery
            new_img[:tY, :tX] = imgTrain

        return new_img

    def save(self, img) -> None:
        cv2.imwrite(
            f"output/{self.dir_name}|{self.now}|n={self.n}|s={self.s}.jpg", img)


if __name__ == "__main__":
    args = parser.parse_args()
    assert (args.dir)

    app = App(args, port=AngleModeDriver()) if args.angle else App(args)
    img_paths = app.generateListOfImgPath()
    output = compressInputImg(img_paths[0])
    for idx, path in enumerate(img_paths[1:]):
        print(idx)
        t_img = compressInputImg(path)
        q_key_point, t_key_point, matches = fetchMatches(output, t_img)
        output = app.mosaic(output, q_key_point, t_img, t_key_point, matches)

    # if args.angle:
    #     drawMatches(output, q_key_point, t_img,
    #                 t_key_point, app.calculate_port.matches, app.dir_name)
    # else:
    #     drawMatches(output, q_key_point, t_img,
    #                 t_key_point, matches, app.dir_name)
    app.save(output)
    exit()
