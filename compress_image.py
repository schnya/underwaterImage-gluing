import cv2


def imgEncodeDecode(filename, quality=50):
    """
    入力された画像リストを圧縮する
    [in]  filename: 入力画像ファイルパス
    [in]  quality:  圧縮する品質 (1-100)
    [out] out_imgs: 出力画像リスト
    """

    img = cv2.imread(filename)
    _, _, ch = img.shape
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", img, encode_param)
    if False == result:
        print("could not encode image!")
        exit()

    return cv2.imdecode(encimg, ch)
