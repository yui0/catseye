from PIL import Image
from itertools import product
from random import random
from random import gauss

class Noise:
    def __init__(self, input_image):
        self.input_image = input_image
        self.input_pix = self.input_image.load()
        self.w, self.h = self.input_image.size

    def saltpepper(self, salt=0.05, pepper=0.05):
        output_image = Image.new("RGB", input_image.size)
        output_pix = output_image.load()

        for x, y in product(*map(range, (self.w, self.h))):
            r = random()
            if r < salt:
                output_pix[x, y] = (255, 255, 255)
            elif r > 1 - pepper:
                output_pix[x, y] = (  0,   0,   0)
            else:
                output_pix[x, y] = self.input_pix[x, y]
        return output_image

    def gaussian(self, amp=10):
        output_image = Image.new("RGB", input_image.size)
        output_pix = output_image.load()
        for x, y in product(*map(xrange, (self.w, self.h))):
            noised_colors = map(lambda x: gauss(x, amp), self.input_pix[x, y])
            noised_colors = map(lambda x: max(0, x), map(lambda x: min(255, x), noised_colors))
            noised_colors = tuple(map(int, noised_colors))
            output_pix[x, y] = noised_colors
        return output_image

class Filter:
    k = tuple([(1/9, 1/9, 1/9) for _ in range(3)])
    gk = (( 1 /16, 1 / 8, 1 /16),
          ( 1 / 8, 1 / 4, 1 /8 ),
          ( 1 /16, 1 / 8, 1 /16))

    def __init__(self, input_image):
        self.input_image = input_image
        self.input_pix = self.input_image.load()
        self.w, self.h = self.input_image.size

    def mean(self, kernel=k):
        # カーネルの総和は1である必要がある
        summation = sum([sum(k) for k in kernel])
        if summation != 1:
            return None

        output_image = Image.new("RGB", input_image.size)
        output_pix = output_image.load()

        # あるピクセルの周辺を走査するための一時変数（きたない）
        W = int(len(kernel[0])/2)
        W = (-W, W+1)
        H = int(len(kernel)/2)
        H = (-H, H+1)

        # 配列の境界判定
        borderX = {0: (0, 2), self.w-1: (-1, 1)}
        borderY = {0: (0, 2), self.h-1: (-1, 1)}

        for y, x in product(*map(xrange, (self.h, self.w))):
            weight_pixels = []
            _W, _H = W, H

            if not 0 < x < self.w - 1:
                _W = borderX[x]
            if not 0 < y < self.h - 1:
                _H = borderY[y]

            for i, j in product(enumerate(range(*_W)), enumerate(range(*_H))):
                # i => (kernelのindex, xからの相対位置)
                # j => (kernelのindex, yからの相対位置)
                i, dx = i
                j, dy = j

                # 重み付き色の計算
                weight_pixel = map(lambda x: x * kernel[i][j], self.input_pix[x+dx, y+dy])
                weight_pixels.append(weight_pixel)

            color = tuple(map(int, map(sum, zip(*weight_pixels))))
            output_pix[x, y] = color
        return output_image

    def gaussian(self, gauss_kernel=gk):
        return self.mean(gauss_kernel)

    def median(self):
        output_image = Image.new("RGB", input_image.size)
        output_pix = output_image.load()

        # ピクセルの周辺走査用変数
        W = (-1, 2)
        H = (-1, 2)

        # 配列の境界判定
        borderX = {0: (0, 2), self.w-1: (-1, 1)}
        borderY = {0: (0, 2), self.h-1: (-1, 1)}

        for y, x in product(*map(xrange, (self.h, self.w))):
            pixels = []
            _W, _H = W, H

            if not 0 < x < self.w - 1:
                _W = borderX[x]
            if not 0 < y < self.h - 1:
                _H = borderY[y]

            # x, y周辺の輝度を得る
            for dx, dy in product(range(*_W), range(*_H)):
                pixel = self.input_pix[x+dx, y+dy]
                pixels.append(pixel)

            # 輝度配列の長さ
            pixlen = len(pixels)

            # 色成分ごとに配列をまとめる
            pixels = zip(*pixels)

            # 輝度順にソートする
            sorted_pixels = map(sorted, pixels)

            # 中央値を得る
            color = tuple(map(lambda x: x[int(pixlen/2)], sorted_pixels))
            output_pix[x, y] = color
        return output_image

def PSNR(original, compare):
    from math import log10, sqrt
    MAX = 2**8 - 1

    w, h = original.size
    opix = original.load()
    cpix = compare.load()

    color_sum = 0
    for x, y in product(*map(range, (w, h))):
        color = zip(*(opix[x, y], cpix[x, y]))
        color = map(lambda x: (x[0] - x[1])**2, color)
        color_sum += sum(color)
    MSE = color_sum/3/w/h
    PSNR = 20 * log10(MAX/sqrt(MSE))
    return PSNR
