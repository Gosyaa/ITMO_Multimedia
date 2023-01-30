import math
from skimage import io
import numpy as np

def gauss(sig, x, y):
    a = 1 / (2 * sig * sig * math.pi)
    b = math.exp((-x * x - y * y) / (2 * sig * sig))
    return (a * b)

def gauss_pir(img, sig, l):
    k = round(6 * sig + 1)
    g = [gauss(sig, x, y) for y in range(-(k // 2), k // 2 + 1) for x in range(-(k // 2), k // 2 + 1)]
    g = np.array(g).reshape(k, k)
    g = g/g.sum()
    ans = [img.copy()]
    for q in range(l):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k1 = 0
                sum = 0
                u1 = 0
                for c in range(i - (k // 2), i + (k // 2) + 1):
                    u1 = 0
                    for u in range(j - (k // 2), j + (k // 2) + 1):
                        if c < 0:
                            k2 = -c + 1
                        elif c >= img.shape[0]:
                            k2 = 2 * img.shape[0] - 1 - c
                        else:
                            k2 = c
                        if u < 0:
                            u2 = -u + 1
                        elif u >= img.shape[1]:
                            u2 = 2 * img.shape[1] - 1 - u
                        else:
                            u2 = u
                        q = int(img[k2, u2]) * g[k1, u1]
                        sum += q
                        u1 += 1
                    k1 += 1
                img[i, j] = sum
        img = img.astype('uint8')
        ans.append((img.copy()))
    return ans

pic1 = io.imread('a.png')
pic2 = io.imread('b.png')
mask = io.imread('mask.png')
mask = 0.299 * mask[:, :, 0] + 0.587 * mask[:, :, 1] + 0.114 * mask[:, :, 2]
pic1 = 0.299 * pic1[:, :, 0] + 0.587 * pic1[:, :, 1] + 0.114 * pic1[:, :, 2]
pic1 = pic1.astype('uint8')
pic2 = 0.299 * pic2[:, :, 0] + 0.587 * pic2[:, :, 1] + 0.114 * pic2[:, :, 2]
pic2 = pic2.astype('uint8')
io.imshow(mask, cmap='gray')
io.show()
io.imshow(pic2, cmap='gray')
io.show()
g_pir1 = gauss_pir(pic1.copy(), 0.33, 3)
l_pir1 = []
g_pir2 = gauss_pir(pic2.copy(), 0.33, 3)
l_pir2 = []
for i in range(1, len(g_pir1)):
    cur = g_pir1[i - 1] - g_pir1[i]
    l_pir1.append(cur)
    cur = g_pir2[i - 1] - g_pir2[i]
    l_pir2.append(cur)
l_pir1.append(g_pir1[-1])
l_pir2.append(g_pir2[-1])
m = gauss_pir(mask.copy(), 0.33, 4)
m = m[1:]
ans = []
test1 = l_pir1[0]
test2 = l_pir2[0]
for i in range(4):
    m[i] = (m[i] > 128).astype('uint')
    cur1 = l_pir1[i] * m[i];
    q = 1 - m[i]
    cur2 = l_pir2[i] * (1 - m[i])
    cur = cur1 + cur2
    cur = cur.astype('uint8')
    ans.append(cur)
out = ans[0]
for i in range(1, 4):
    out = out + ans[i]
    test1 += l_pir1[i]
    test2 += l_pir2[i]
out = np.clip(out, 0, 255)
io.imsave('out.png', out)
io.imshow(out, cmap='gray')
io.show()