# -*- coding: utf-8 -*-
import os, sys, random
from PIL import Image, ImageFilter

def crop_random(img, sx, sy):
    x = random.randint(0, img.width -sx)
    y = random.randint(0, img.height -sy)
    return img.crop((x, y, x+sx, y+sy))

def main():
    data_dir_path = u"./out/"
    data_dir_path_in = u"./in/"
    file_list = os.listdir(r'./in/')

    width = 256
    height = 256
    usage = 'Usage: python {} FILE [--verbose] [--cat <file>] [--help]'\
            .format(__file__)
    args = sys.argv
    '''
    if len(arguments) == 1:
        return usage
    # ファイル自身を指す最初の引数を除去
    arguments.pop(0)
    # 引数として与えられたfile名
    fname = arguments[0]
    if fname.startswith('-'):
        return usage
    # - で始まるoption
    options = [option for option in arguments if option.startswith('-')]

    if '-h' in options or '--help' in options:
        return usage
    '''

    for file_name in file_list:
        root, ext = os.path.splitext(file_name)

        if ext == u'.png' or u'.jpeg' or u'.jpg':
            img = Image.open(data_dir_path_in + '/' + file_name)
            if img.width > img.height:
                img = img.resize((int(img.width * (float(height) / img.height)), height), Image.BICUBIC)
            else:
                img = img.resize((width, int(img.height * (float(width) / img.width))), Image.BICUBIC)
            tmp = crop_random(img, 224, 224)
            tmp.save(data_dir_path + '/' + root +'_r31.jpg')

            tmp = img.transpose(Image.FLIP_LEFT_RIGHT)
            tmp.save(data_dir_path + '/' + root +'_r01.jpg')
            tmp = img.transpose(Image.FLIP_TOP_BOTTOM)
            tmp.save(data_dir_path + '/' + root +'_r02.jpg')
            tmp = img.transpose(Image.ROTATE_90)
            tmp.save(data_dir_path + '/' + root +'_r03.jpg')
            tmp = img.transpose(Image.ROTATE_180)
            tmp.save(data_dir_path + '/' + root +'_r04.jpg')
            tmp = img.transpose(Image.ROTATE_270)
            tmp.save(data_dir_path + '/' + root +'_r05.jpg')
            tmp = img.rotate(15)
            tmp.save(data_dir_path + '/' + root +'_r06.jpg')
            tmp = img.rotate(30)
            tmp.save(data_dir_path + '/' + root +'_r07.jpg')
            tmp = img.rotate(45)
            tmp.save(data_dir_path + '/' + root +'_r08.jpg')
            tmp = img.rotate(60)
            tmp.save(data_dir_path + '/' + root +'_r09.jpg')
            tmp = img.rotate(75)
            tmp.save(data_dir_path + '/' + root +'_r10.jpg')
            tmp = img.rotate(105)
            tmp.save(data_dir_path + '/' + root +'_r11.jpg')
            tmp = img.rotate(120)
            tmp.save(data_dir_path + '/' + root +'_r12.jpg')
            tmp = img.rotate(135)
            tmp.save(data_dir_path + '/' + root +'_r13.jpg')
            tmp = img.rotate(150)
            tmp.save(data_dir_path + '/' + root +'_r14.jpg')
            tmp = img.rotate(165)
            tmp.save(data_dir_path + '/' + root +'_r15.jpg')
            tmp = img.rotate(195)
            tmp.save(data_dir_path + '/' + root +'_r16.jpg')
            tmp = img.rotate(210)
            tmp.save(data_dir_path + '/' + root +'_r17.jpg')
            tmp = img.rotate(225)
            tmp.save(data_dir_path + '/' + root +'_r18.jpg')
            tmp = img.rotate(240)
            tmp.save(data_dir_path + '/' + root +'_r19.jpg')
            tmp = img.rotate(255)
            tmp.save(data_dir_path + '/' + root +'_r20.jpg')
            tmp = img.rotate(285)
            tmp.save(data_dir_path + '/' + root +'_r21.jpg')
            tmp = img.rotate(300)
            tmp.save(data_dir_path + '/' + root +'_r22.jpg')
            tmp = img.rotate(315)
            tmp.save(data_dir_path + '/' + root +'_r23.jpg')
            tmp = img.rotate(330)
            tmp.save(data_dir_path + '/' + root +'_r24.jpg')
            tmp = img.rotate(345)
            tmp.save(data_dir_path + '/' + root +'_r25.jpg')
            tmp = img.filter(ImageFilter.FIND_EDGES)
            tmp.save(data_dir_path + '/' + root +'_r26.jpg')
            tmp = img.filter(ImageFilter.EDGE_ENHANCE)
            tmp.save(data_dir_path + '/' + root +'_r27.jpg')
            tmp = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            tmp.save(data_dir_path + '/' + root +'_r28.jpg')
            tmp = img.filter(ImageFilter.UnsharpMask(radius=5, percent=150, threshold=2))
            tmp.save(data_dir_path + '/' + root +'_r29.jpg')
            tmp = img.filter(ImageFilter.UnsharpMask(radius=10, percent=200, threshold=5))
            tmp.save(data_dir_path + '/' + root +'_r30.jpg')

if __name__ == '__main__':
    main()
