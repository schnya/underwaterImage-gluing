import argparse
import glob
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from drawMatches import drawMatches
from fetchMatches import fetchMatches

from imageCompression import compressInputImg

sift = cv2.SIFT.create()
bf = cv2.BFMatcher()
bgr2rgb = cv2.COLOR_BGR2RGB
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="ディレクトリ名")
parser.add_argument("-n", "--count", type=int, help="使う画像の枚数")

sns.set()

"""
cv2.DMatch
    distance
        特徴量記述子間の距離．類似度? 低いほど良い
    trainIdx
        学習記述子（参照データ）の記述子のインデックス
    queryIdx
        クエリ記述子（検索データ）の記述子のインデックス

cv2.KeyPoint
    angle
        計算されたキーポイントの方向（適用しない場合は-1)。
        これは [0,360) 度の単位で，画像座標系を基準として、つまり時計回りに測られます。
    octave
        キーポイントが抽出されたオクターブ（ピラミッド層
    pt
        キーポイントの座標
    response
        最も強力なキーポイントが選択された応答です。
        さらにソートやサブサンプリングに使用することができます。
    size
        意味のあるキーポイント近傍の直径
"""

COLUMNS = ["x", "y", "degree"]


def obtainMode(input_df, col: str):
    output_df = input_df.copy()
    mode = output_df[col].mode()[0]
    print("角度の最頻値:", mode)
    # 外れ値基準の下限・上限を取得
    bottom = mode - 5
    up = mode + 5
    output_df = output_df[col][(bottom <= output_df[col]) & (output_df[col] <= up)]
    return output_df.index

def getDegree(y: float, x: float, y2: float, x2: float) -> float:
    a = np.array([y, x])
    b = np.array([y2, x2])
    vec = b - a

    # 逆正接; 返り値は-piからpi（-180度から180度）の間
    return np.degrees(np.arctan2(vec[0], vec[1]))
    


def caluculate(m, q_kp, t_kp):
    qx, qy = q_kp[m[0].queryIdx].pt
    qx, qy = int(qx), int(qy)
    tx, ty = t_kp[m[0].trainIdx].pt
    tx, ty = int(tx), int(ty)
    x = qx - tx
    y = qy - ty
    degree = getDegree(qy, qx, ty, tx)

    return x, y, int(degree)


def collage(q_kp, t_kp, matches):
    x, y = 0, 0
    for m in matches[:20]:
        x, y, _ = caluculate(m, q_kp, t_kp)
    x, y = int(x / min(len(matches), 20)), int(y / min(len(matches), 20))

    print("移動距離 x:", x, "y:", y)


def makeDataFrame(q_kp, t_kp, matches, after: bool = False):
    now = datetime.now().strftime("%m%d_%H時%M")
    df = pd.DataFrame(columns=COLUMNS)

    for m in matches:
        x, y, degree = caluculate(m, q_kp, t_kp)
        s = pd.Series([x, y, degree])
        df = pd.DataFrame(np.vstack([df.values, s.values]), columns=df.columns)

    # sum_x, sum_y = df["x"].sum(), df["y"].sum()
    # print("移動距離 x:", sum_x / len(matches), "y:", sum_y / len(matches))

    df["degree"].hist()
    plt.show()

    sns.pairplot(df).savefig(
        f"output/p{now}airplot{'_after' if after else ''}|{dir_name}.png"
    )
    # sns.displot(df.degree).savefig(
    #     f"output/d{dir_name}isplot_degree{'_after' if after else ''}|{datetime.now()}.png"
    # )
    # sns.boxplot(df.y).get_figure().savefig("output/boxplot_y.png")
    # index = obtainMode(df, "degree").index
    # sns.displot(output_x).savefig("output/displot_x-detected.png")
    # sns.displot(output_y).savefig("output/displot_y-detected.png")
    # sns.boxplot(output_x).get_figure().savefig("output/boxplot_x-detected.png")
    # sns.boxplot(output_y).get_figure().savefig("output/boxplot_y-detected.png")

    return df


if __name__ == "__main__":
    args = parser.parse_args()
    dir_name = args.dir
    n = args.count
    today = datetime.now().isoformat().split("T")[0]
    filenames = sorted(glob.glob(f"{dir_name}/*.JPG"), reverse=True)[:n]

    output = compressInputImg(filenames[0])
    for name in filenames[1:]:
        qKeyPoints, train_img, tKeyPoints, matches = fetchMatches(output, name)
        print("何もしなかった場合")
        print("要素数:", len(train_img))
        collage(tKeyPoints, matches, train_img)
        drawMatches(output, tKeyPoints, qKeyPoints, matches, train_img, dir_name)
        df = makeDataFrame(tKeyPoints, matches, train_img)

        print("- - - - - - - - - - - - - -")
        print("角度の最頻値±5°までを取得した場合")
        index = obtainMode(df, "degree")
        print("要素数:", len(index))
        _matches = [train_img[i] for i in index]
        makeDataFrame(tKeyPoints, matches, _matches, after=True)

        drawMatches(output, tKeyPoints, qKeyPoints, matches, _matches, dir_name)
