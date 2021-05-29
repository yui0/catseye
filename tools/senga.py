import os
import sys
import glob
import cv2
import numpy as np

def make_contour_image(path):
    print(path)
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("senga_gray.jpg", gray)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    cv2.imwrite("senga_dilated.jpg", dilated)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)
    cv2.imwrite("senga_diff.jpg", diff)

    # 白黒反転
    contour = 255 - diff
    print('senga/'+path)
    cv2.imwrite('senga/'+os.path.basename(path), contour)
    return contour

if __name__ == '__main__':
    if len(sys.argv) == 2:
        lists = glob.glob(sys.argv[1] + '/*')
        lists += glob.glob(sys.argv[1] + '/**/*')
        print(lists)
        os.makedirs('senga', exist_ok=True)
        for f in lists:
            make_contour_image(f)
    else:
        print('arguments error')
