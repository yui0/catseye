import os
import sys
import cv2
from PIL import Image
 
 
SPINS = {
    'flipTB': Image.FLIP_TOP_BOTTOM,
    'spin90': Image.ROTATE_90,
    'spin270': Image.ROTATE_270,
    'flipLR': Image.FLIP_LEFT_RIGHT,
}
 
 
def filter_extension(img_dir, extensions):
    """ある拡張子のファイルだけを返す"""
 
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.split('.')[-1] in extensions:
                yield os.path.join(root, file), root, file
 
 
def resize_img(img_dir, extensions, size):
    """ディレクトリの画像を、リサイズし上書きする"""
 
    for file_path, _, _ in filter_extension(img_dir, extensions):
        img = Image.open(file_path)
        new_img = img.resize(size)
        new_img.save(file_path)
 
 
def spin_img(img_dir, extensions):
    """ディレクトリの画像を、4方向にスピンしたものを追加する"""
 
    for file_path, root, file in filter_extension(img_dir, extensions):
        img = Image.open(file_path)
        for name, kind in SPINS.items():
            new_img = img.transpose(kind)
            new_file_name = '{0}_{1}'.format(name, file)
            new_file_path = os.path.join(root, new_file_name)
            new_img.save(new_file_path)
 
 
def detect_face(image, cascade_path):
    """顔の検出を行う"""
 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
 
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(
        image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
 
    return facerect
 
 
def collect_img(cap, fps, extension, save_dir, cascade_path):
    """動画から、顔を検出し保存する"""
 
    frame_number = 0
    img_number = 0
 
    while(cap.isOpened()):
        frame_number += 1
        ret, image = cap.read()
        if not ret:
            break
 
        # ループn回毎に処理に入る。少なくすると画像が増える（細かくキャプチャされる）
        if frame_number % fps == 0:
            facerect = detect_face(image, cascade_path)
 
            # 認識結果がnullだったら次のframeへ
            if len(facerect) == 0:
                continue
 
            for rect in facerect:
                croped = image[
                    rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                file_name = '{0}.{1}'.format(format(img_number, '0>8'), extension)
                path = os.path.join(save_dir, file_name)
                cv2.imwrite(path, croped)
                img_number += 1
    return img_number
 
 
def start(video_path, cascade_path, save_dir='img', extension='png',
          resize_size=(50, 50), fps=50,  resize=True, pad=True,):
    """動画から、画像の取得を開始する"""
 
    # 保存先のディレクトリが存在しなければ作る
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
 
    # 動画から、画像を抜き出す
    cap = cv2.VideoCapture(video_path)
    img_numbers = collect_img(cap, fps, extension, save_dir, cascade_path)
    cap.release()
    print('画像のキャプチャを終了', img_numbers)
 
    # 画像を、リサイズし統一する
    if resize:
        resize_img(save_dir, (extension,), resize_size)
        print('リサイズの終了')
 
    # スピン画像を追加し、データを水増しする
    if pad:
        spin_img(save_dir, (extension,))
        print('スピン画像の追加を終了')
 
if __name__ == '__main__':
    #start('video.mp4', 'lbpcascade_animeface.xml')
    # start('kemono.mp4', 'lbpcascade_animeface.xml',
    #      save_dir='kemono', fps=100, extension='jpg')
    start(sys.argv[1], 'haarcascade_frontalface_alt.xml', fps=20, resize=False, pad=False)

