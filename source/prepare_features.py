import os
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import normalize

RAW_DATA_PATHS = [
    r'../dataset/InertialGaitData/AutomaticExtractionData_IMUZCenter',
    # r'../dataset/InertialGaitData/ManualExtractionData/Android',
    r'../dataset/InertialGaitData/ManualExtractionData/IMUZCenter',
    r'../dataset/InertialGaitData/ManualExtractionData/IMUZLeft',
    r'../dataset/InertialGaitData/ManualExtractionData/IMUZRight'
]

WINDOW_LENGTH = 100
N_USERS = 17
SAVEPATH = r'../features'
NORMALIZE = True


def get_magnitude(data):
    x_2 = data[:, 0] * data[:, 0]
    y_2 = data[:, 1] * data[:, 1]
    z_2 = data[:, 2] * data[:, 2]
    m_2 = x_2 + y_2 + z_2
    m = np.sqrt(m_2)
    return np.reshape(m, (m.shape[0], 1))


def get_window(data):
    start = 0
    size = data.shape[0]
    while start < size:
        end = start + WINDOW_LENGTH
        yield start, end
        start += int(WINDOW_LENGTH // 2)  # 50% overlap, hence divide by two


def get_summary(data, start, end):
    data_window = data[start:end]
    acf = np.correlate(data_window, data_window, mode='full')
    acv = np.cov(data_window.T, data_window.T)
    sq_err = (data_window - np.mean(data_window)) ** 2
    return [
        np.mean(data_window),
        np.std(data_window),
        np.var(data_window),
        np.min(data_window),
        np.max(data_window),
        np.mean(acf),
        np.std(acf),
        np.mean(acv),
        np.std(acv),
        skew(data_window),
        kurtosis(data_window),
        math.sqrt(np.mean(sq_err))
    ]


def get_features(raw_data):
    data_parts = [None] * 2
    data_parts[0], data_parts[1] = raw_data[:, :3], raw_data[:, 3:]
    features = None
    for data in data_parts:
        data = np.concatenate((data, get_magnitude(data)), axis=1)

        for (start, end) in get_window(data):
            window_features = []
            for j in range(data.shape[1]):
                window_features += get_summary(data[:, j], start, end)
            if features is None:
                features = np.array(window_features)
            features = np.vstack((features, np.array(window_features)))
            print(features.shape)
            exit(0)
    return features


if __name__ == '__main__':

    final_features = None
    s_time = time.time()
    for dir in RAW_DATA_PATHS:
        for filename in os.listdir(dir):
            filepath = dir + os.sep + filename
            print(filepath)
            user_id = int(filename.split('_')[1].replace('ID', ''))  # Filename: T0_ID000104_Center_seq0.csv

            data = np.genfromtxt(filepath, delimiter=',', skip_header=2)
            features = get_features(data)
            print(data)
            exit(0)

            left_column = np.empty((features.shape[0], 1))
            left_column.fill(int(user_id))
            # print(left_column.shape)
            features = np.hstack((left_column, features))
            if final_features is None:
                final_features = features
            else:
                final_features = np.vstack((final_features, features))
            print('User ID: ', user_id, 'Feature Shape: ', features.shape)

            # visited[user_id] = True

    print('Feature Generation Time:', time.time() - s_time)
    print(final_features.shape)
    np.savetxt(SAVEPATH + os.sep + 'allinone_features.csv', final_features, delimiter=',')
