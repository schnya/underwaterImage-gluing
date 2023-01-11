import numpy as np

from port.port import Port


class MeanDriver(Port):
    def measureDistanceTraveledByKeyPoint(self, matches, q_kp, t_kp, K=50) -> tuple[int, int]:
        x, y = 0, 0
        for m in matches[:K]:
            m = m[0]

            px, py = q_kp[m.queryIdx].pt
            px, py = int(px), int(py)

            nx, ny = t_kp[m.trainIdx].pt
            nx, ny = int(nx), int(ny)

            x += px - nx
            y += py - ny
        x, y = int(x / K), int(y / K)
        print('y:', y, 'x:', x)
        return x, y

    def createEmptyImage(self, base_height, base_width, overlay_height, overlay_width, x, y):
        height = max([base_height, base_height - y, abs(y) + overlay_height])
        width = max([base_width, base_width - x, abs(x) + overlay_width])
        imgArr = np.zeros((height, width, 3), np.uint8)
        print('🆕 Image Size:', imgArr.shape)
        return imgArr


if __name__ == "__main__":
    driver = MeanDriver()

    bH, bW = 4050, 3100
    oH, oW = 4000, 3000
    y, x = -200, 15
    new_img = driver.createEmptyImage(bH, bW, oH, oW, x, y)
    height, width, _ = new_img.shape
    assert (height == 4250 and width == 3100)

    bH, bW = 5050, 4100
    oH, oW = 4000, 3000
    y, x = -200, 15
    new_img = driver.createEmptyImage(bH, bW, oH, oW, x, y)
    height, width, _ = new_img.shape
    assert (height == 5250 and width == 4100)

    bH, bW = 5050, 4100
    oH, oW = 4000, 3000
    y, x = -2200, -1500
    new_img = driver.createEmptyImage(bH, bW, oH, oW, x, y)
    height, width, _ = new_img.shape
    assert (height == 7250 and width == 5600)
