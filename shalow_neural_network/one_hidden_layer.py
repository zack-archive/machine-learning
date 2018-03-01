from load_data import get_data
import numpy as np

train, train_label, test, test_label = get_data()


def sg(X):
    return 1 / (1 + np.exp(-X))


def drelu(X):
    return np.where(X > 0, 1, 0)


if __name__ == "__main__":
    rate = 0.015
    h = int(input("Please input the number of neurons in the first layer!"))
    np.random.seed(5)
    w1 = np.random.random((h, 3))*0.01
    b1 = np.random.random((h, 1))
    w2 = np.random.random((1, h))*0.01
    b2 = 0
    for i in range(100000):
        z1 = np.dot(w1, train) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(w2, a1) + b2
        a2 = sg(z2)
        loss = -np.dot(train_label, np.log(a2).T)-np.dot(1-train_label, np.log(1-a2).T)
        dz2 = a2 - train_label
        dw2 = np.dot(dz2, a1.T) / 524
        db2 = np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(w2.T, dz2) * drelu(z1)
        dw1 = np.dot(dz1, train.T) / 524
        db1 = np.sum(dz1, axis=1, keepdims=True)
        w1 = w1 - rate * dw1
        b1 = b1 - rate * db1
        w2 = w2 - rate * dw2
        b2 = b2 - rate * db2

    tz1 = np.dot(w1, test)+b1
    ta1 = np.maximum(tz1, 0)
    tz2 = np.dot(w2, ta1) + b2
    ta2 = sg(tz2)
    accuracy = 1-np.sum(np.fabs(ta2 - test_label)) / 175  # 与标准label相减后求绝对值就是错误预测的数量
    print(accuracy)