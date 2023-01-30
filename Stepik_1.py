import math
import skimage.color
from skimage import io
from scipy import  signal
import numpy as np

def gauss(sig, x, y):
    a = 1 / (2 * sig * sig * math.pi)
    b = math.exp((-x * x - y * y) / (2 * sig * sig))
    return (a * b)

def kernel(sigma):
    k = round(sigma * 6 + 1)
    filt = np.zeros((k, k))
    for i in range(-(k // 2), k // 2 + 1):
        for j in range(-(k // 2), k // 2 + 1):
            filt[i + k // 2, j + k // 2] = gauss(sigma, i, j)
    return filt / filt.sum()

def gauss_pir(img, sigma, n_layers):
    ker = kernel(sigma)
    pir = [img]
    for i in range(n_layers):
        img = signal.convolve2d(img, ker, mode='same', boundary='symm')
        img = np.clip(img, 0, 1)
        pir.append(img)
    return pir

def lap_pir(img, sigma, n_layers):
    g = gauss_pir(img.copy(), sigma, n_layers)
    pir = []
    for i in range(n_layers):
        pir.append(g[i] - g[i + 1])
    pir[-1] = g[-1]
    return pir

def main(img1, img2, mask):
    sigma = 4
    n_layers = 10
    l_pir1 = lap_pir(img1.copy(), sigma, n_layers)
    l_pir2 = lap_pir(img2.copy(), sigma, n_layers)
    g_mask = gauss_pir(mask.copy(), sigma, n_layers)
    ans = []
    for i in range(n_layers):
        cur = l_pir1[i] * g_mask[i + 1] + l_pir2[i] * (1 - g_mask[i + 1])
        ans.append(cur)
    return ans

img1 = io.imread('a-gray.png')
img1 = skimage.img_as_float(img1)
img2 = io.imread('b-gray.png')
img2 = skimage.img_as_float(img2)
m = io.imread('m-gray.png')
m = skimage.img_as_float(m)
ans = main(img1, img2, m)
out = ans[0]
for i in range(1, len(ans)):
    out += ans[i]
out = np.clip(out, 0, 1)
io.imshow(out, cmap='gray')
io.show()
io.imsave('out.png', out)