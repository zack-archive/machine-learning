import numpy as np
import csv


def get_data():
    train_csv = "breast-cancer-train.csv"
    test_csv = "breast-cancer-test.csv"
    with open(train_csv) as file:  # 使用with结构能避免在程序出错时，由文件不能关闭导致的文件损坏
        reader = csv.reader(file)
        train = []
        train_label = []
        next(reader)  # 跳过第一行（信息行）
        for row in reader:
            train.append(row[0:3])  # 取得前三列（features）
            train_label.append([row[3]])  # labels
        train = np.array(train, float).T
        train_label = np.array(train_label, float).T
    with open(test_csv) as file:
        reader = csv.reader(file)
        test = []
        test_label = []
        next(reader)
        for row in reader:
            test.append(row[0:3])
            test_label.append(row[3])
        test = np.array(test, float).T
        test_label = np.array(test_label, float).T
    assert(train.shape == (3, 524))
    assert(test.shape == (3, 175))
    return train, train_label, test, test_label


def feature_scaling(m):  # 把数据进行feature_scaling之后，准确率下降，未知原因
    mean = np.sum(m, axis=1, keepdims=True)/(m.shape[1])
    m_sd_0 = np.max(m[0]) - np.min(m[0])
    m_sd_1 = np.max(m[1]) - np.min(m[1])
    m_sd_2 = np.max(m[2]) - np.min(m[2])
    m_sd = [[m_sd_0], [m_sd_1], [m_sd_2]]
    m = (m - mean)/m_sd
    return m