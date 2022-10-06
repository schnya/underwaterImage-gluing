from PIL import Image
from io import BytesIO
import glob
import os

# コンフィグ
COMPRESS_QUALITY = 50  # 圧縮のクオリティ

# JPEG形式とPNG形式の画像ファイルを用意
dir = "2019_10_01 Aw 1/"
files = glob.glob(dir + "PA*")
for file_path in files:
    #############################
    #     JPEG形式の圧縮処理     #
    #############################
    # ファイル名を取得
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as inputfile:
        # バイナリモードファイルをPILイメージで取得
        im = Image.open(inputfile)
        # JPEG形式の圧縮を実行
        im_io = BytesIO()
        im.save(im_io, "jpeg", quality=COMPRESS_QUALITY)
    with open(dir + "comp_" + filename, mode="wb") as outputfile:
        # 出力ファイル(comp_png_image.png)に書き込み
        outputfile.write(im_io.getvalue())
