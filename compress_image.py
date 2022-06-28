from PIL import Image
from io import BytesIO
import os

# コンフィグ
COMPRESS_QUALITY = 50  # 圧縮のクオリティ

# JPEG形式とPNG形式の画像ファイルを用意
dir = "OneDrive_1_6-28-2022/"
ImageList = ["PA011324.JPG", "PA011325.JPG", "PA011326.JPG"]

#############################
#     JPEG形式の圧縮処理     #
#############################
for img in ImageList:
    # ファイル名を取得
    file_name = os.path.splitext(os.path.basename(dir + img))[0]
    with open(dir + img, "rb") as inputfile:
        # バイナリモードファイルをPILイメージで取得
        im = Image.open(inputfile)
        # JPEG形式の圧縮を実行
        im_io = BytesIO()
        im.save(im_io, "jpeg", quality=COMPRESS_QUALITY)
    with open(dir + "comp_" + file_name + ".jpg", mode="wb") as outputfile:
        # 出力ファイル(comp_png_image.png)に書き込み
        outputfile.write(im_io.getvalue())
