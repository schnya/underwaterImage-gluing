from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from port.port import Port

COLUMNS = ["x", "y", "degree"]
def now(): return datetime.now().strftime("%m%d_%H時%M")


class AngleModeDriver(Port):
    def _measureAngleMode(self, df, matches, rad=5):
        # 外れ値基準の下限・上限を取得
        output_df = df.copy()
        degree = output_df["degree"]

        mode = degree.mode()[0]
        bottom, up = mode - rad, mode + rad
        # print("角度の最頻値:", mode)

        output_df = degree[(bottom <= degree) & (degree <= up)]
        index = output_df.index

        # print("要素数:", len(index))
        return [matches[i] for i in index]

    def _saveFig(self, df):
        sns.displot(df.degree).savefig(
            f"output/displot_degree|{datetime.now()}.png"
        )
        plt.close()

    def _hoge(self, matches, q_kp, t_kp, K=50):
        df = pd.DataFrame(columns=COLUMNS)
        for m in matches[:K]:
            m = m[0]

            px, py = q_kp[m.queryIdx].pt
            px, py = int(px), int(py)

            nx, ny = t_kp[m.trainIdx].pt
            nx, ny = int(nx), int(ny)

            x, y = px - nx, py - ny
            degree = int(np.degrees(np.arctan2(x, y)))

            s = pd.Series([x, y, degree])
            df = pd.DataFrame(
                np.vstack([df.values, s.values]), columns=df.columns)

        return df

    def measureDistanceTraveledByKeyPoint(self, matches, q_kp, t_kp) -> tuple[int, int]:
        df = self._hoge(matches, q_kp, t_kp)
        # self._saveFig(df)
        # x, y = int(df["x"].mean()), int(df["y"].mean())
        # print('全部の平均 y:', y, 'x:', x)
        
        self.matches = self._measureAngleMode(df, matches)
        df = self._hoge(self.matches, q_kp, t_kp)
        # self._saveFig(df)

        x, y = int(df["x"].mean()), int(df["y"].mean())
        # print('最頻値に絞った平均 y:', y, 'x:', x)
        return x, y

    def createEmptyImage(self, base_height, base_width, overlay_height, overlay_width, x, y):
        height = max([base_height, base_height - y, abs(y) + overlay_height])
        width = max([base_width, base_width - x, abs(x) + overlay_width])
        imgArr = np.zeros((height, width, 3), np.uint8)
        # print('🆕 Image Size:', imgArr.shape)
        return imgArr


if __name__ == "__main__":
    x, y = 50 - 10, 100 - 10
    vec = np.array([50, 100]) - np.array([10, 10])
    assert (np.arctan2(vec[0], vec[1]) == np.arctan2(40, 90))
