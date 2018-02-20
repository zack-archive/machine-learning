# coding: utf-8
import numpy as np
import time
from load_data import get_data


train, train_label, test, test_label = get_data()
np.random.seed(1)
w = np.random.random((3, 1))
b = 0
rate = 0.05  # 这是learning rate
start = time.clock()
for i in range(300000):  # 200000次iteration
    # forward propagation
    z = np.dot(w.T, train) + b
    a = 1 / (1 + np.exp(-z))
    # 确定两者shape相同
    assert (a.shape == train_label.shape)
    # backpropagation
    dz = a - train_label
    db = np.sum(dz) / 524
    dw = np.dot(train, dz.T) / 524
    w = w - rate * dw
    b = b - rate * db
print(w, b)
z_test = np.dot(w.T, test) + b
a_test = 1 / (1 + np.exp(-z_test))  # issue: 会产生warning： overflow in exp, 之前的也有
                                    # 解决方法包括安装bigfloat库， 可是没有成功安装
accuracy = 1 - np.sum(np.fabs(a_test - test_label)) / 175  # 与标准label相减后求绝对值就是错误预测的数量
print("Accuracy: %s" % accuracy)
end = time.clock()
print("time: " + str(end - start))
