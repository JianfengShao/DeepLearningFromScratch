import pickle

import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from PIL import Image


# 阶跃函数
def steep_function(x):
    y = x > 0
    return y.astype(int)


# sigmod 函数
def sigmod(x):
    return 1 / (1 + np.exp(-x))


# ReLU函数(Rectified Linear Unit)
def ReLU(x):
    return np.maximum(0, x)


# 恒等函数
def identity_function(x):
    return x


# Softmax函数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    exp_sum = np.sum(exp_a)
    y = exp_a / exp_sum

    return y


# 画出函数图形
def showActivationFunc():
    x = np.arange(-5.0, 5.0, 0.1)
    y = ReLU(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# 读取并显示Minist图像
def MNIST_show():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    label = t_train[0]
    img = img.reshape(28, 28)
    print(label)
    img_show(img)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)

    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmod(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmod(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:" + str(accuracy_cnt / len(x)))
