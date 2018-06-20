import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def rgb2gray(rgb):
    w, h = len(rgb[0]), len(rgb);
    gray = [[0 for x in range(w)] for y in range(h)]

    for row in range(len(rgb)):
        for col in range(len(rgb[row])):
            acc = 0;
            for colours in range(len(rgb[row][col])):
                acc += rgb[row][col][colours]
            math = acc/3
            gray[row][col] = int(math)
    return(gray)

def generateGaussian(shape=(3,3) ,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def convolve(img, fltr):
    w, h = len(img[0])-2, len(img)-2;
    newImg = [[0 for x in range(w)] for y in range(h)]

    for row in range(int(len(fltr)/2), len(img) - len(fltr) - 1):
        for col in range(int(len(fltr)/2), len(img[row]) - len(fltr) - 1):
            acc = 0
            for fltr_row in range(len(fltr)):
                for fltr_col in range(len(fltr[fltr_row])):
                    acc += img[row + fltr_row][col + fltr_col] * \
                                      fltr[fltr_row][fltr_col]
            newImg[row][col] = acc
    return newImg

def magnitude(Gx, Gy):
    w, h = len(Gx[0]), len(Gx);
    newImg = [[0 for x in range(w)] for y in range(h)]

    for row in range(0, len(Gx)):
        for col in range(0, len(Gx[0])):
            maths = math.sqrt(math.pow(Gx[row][col], 2) + math.pow(Gy[row][col],2))
            newImg[row][col] = int(maths)
    return newImg

dir = "Pictures/"
img_RGB = misc.imread(dir + "room_2.jpg")
gaussian3X3 = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]
sobel_X = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
sobel_Y = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]


img_GREY = rgb2gray(img_RGB)
img_GAUS = convolve(img_GREY, generateGaussian((5,5), 2))
img_Gx = convolve(img_GAUS, sobel_X)
img_Gy = convolve(img_GAUS, sobel_Y)
img_Edge = magnitude(img_Gx, img_Gy)


plt.imshow(img_RGB)
plt.show()

plt.imshow(img_GREY, cmap="gray")
plt.show()

plt.imshow(img_GAUS, cmap = "gray")
plt.show()

plt.imshow(img_Gx, cmap="gray")
plt.show()

plt.imshow(img_Gy, cmap = "gray")
plt.show()

plt.imshow(img_Edge, cmap = "gray")
plt.show()
