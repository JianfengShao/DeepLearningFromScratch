import numpy as np


# 鱼书第二章 感知机

# 与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.7, 0.7])
    b = -0.7
    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


# 异或门
def XOR(x1, x2):
    result1 = OR(x1, x2)
    result2 = NAND(x1, x2)
    return AND(result1, result2)


# 第三章
import matplotlib.pylab as plt


def steep_function(x):
    y = x > 0
    return y.astype(int)


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# img = x_train[0]
# label = t_train[0]
#
# print(label)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)

for i in range(5):
    print(i)
