from load_data import get_data
import numpy as np

train, train_label, test, test_label = get_data()


# forward propagation


def sg(X):
    return 1 / (1 + np.exp(-X))


def drelu(X):
    return np.where(X > 0, 1, 0)


if __name__ == "__main__":
    rate = 0.01
    w1 = np.random.random((20, 3))*0.01
    b1 = np.random.random((20, 1))
    w2 = np.random.random((20, 20))*0.01
    b2 = np.random.random((20, 1))
    w3 = np.random.random((10, 20))*0.01
    b3 = np.random.random((10, 1))
    w4 = np.random.random((5, 10))*0.01
    b4 = np.random.random((5, 1))
    w5 = np.random.random((1, 5))*0.01
    b5 = 0
    for i in range(10000):
        z1 = np.dot(w1, train) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(w2, a1) + b2
        a2 = np.maximum(z2, 0)
        z3 = np.dot(w3, a2) + b3
        a3 = np.maximum(z3, 0)
        z4 = np.dot(w4, a3) + b4
        a4 = np.maximum(z4, 0)
        z5 = np.dot(w5, a4) + b5
        a5 = sg(z5)
        print(i)
        print(-np.dot(train_label, np.log(a5).T)-np.dot(1-train_label, np.log(1-a5).T))
        dz5 = a5 - train_label
        dw5 = np.dot(dz5, a4.T)/524
        db5 = np.sum(dz5, axis=1, keepdims=True)
        dz4 = np.dot(w5.T, dz5) * drelu(z4)
        dw4 = np.dot(dz4, a3.T)/524
        db4 = np.sum(dz4, axis=1, keepdims=True)
        dz3 = np.dot(w4.T, dz4) * drelu(z3)
        dw3 = np.dot(dz3, a2.T)/524
        db3 = np.sum(dz3, axis=1, keepdims=True)
        dz2 = np.dot(w3.T, dz3)*drelu(z2)
        dw2 = np.dot(dz2, a1.T) / 524
        db2 = np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(w2.T, dz2) * drelu(z1)
        dw1 = np.dot(dz1, train.T) / 524
        db1 = np.sum(dz1, axis=1, keepdims=True)
        w1 = w1 - rate * dw1
        b1 = b1 - rate * db1
        w2 = w2 - rate * dw2
        b2 = b2 - rate * db2
        w3 = w3 - rate * dw3
        b3 = b3 - rate * db3
        w4 = w4 - rate * dw4
        b4 = b4 - rate * db4
        w5 = w5 - rate * dw5
        b5 = b5 - rate * db5

    tz1 = np.dot(w1, test)+b1
    ta1 = np.maximum(tz1, 0)
    tz2 = np.dot(w2, ta1) + b2
    ta2 = np.maximum(tz2, 0)
    tz3 = np.dot(w3, ta2) + b3
    ta3 = np.maximum(tz3, 0)
    tz4 = np.dot(w4, ta3) + b4
    ta4 = np.maximum(tz4, 0)
    tz5 = np.dot(w5, ta4) + b5
    ta5 = sg(tz5)
    accuracy = 1-np.sum(np.fabs(ta5 - test_label)) / 175  # 与标准label相减后求绝对值就是错误预测的数量
    print(accuracy)