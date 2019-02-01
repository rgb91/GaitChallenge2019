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
GENDER_AGE_FILE_PATH = r'../dataset/InertialGaitData/IDGenderAgelist.csv'
GENDER_AGE_DICT = {}
WINDOW_LENGTH = 200
SAVEPATH = r'../features'


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
    # print('In summary: ', data.shape)
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
    final_features = None
    for data in data_parts:
        data = np.concatenate((data, get_magnitude(data)), axis=1)
        features = None
        for (start, end) in get_window(data):
            window_features = []
            for j in range(data.shape[1]):
                window_features += get_summary(data[:, j], start, end)
            if features is None:
                features = np.array(window_features)
            else:
                features = np.vstack((features, np.array(window_features)))
        if final_features is None:
            final_features = np.array(features)
        else:
            final_features = np.hstack((final_features, features))

    if (len(final_features.shape)) < 2:
        final_features = final_features.reshape(1, -1)
    return final_features


def load_gender_age():
    g_a_data = np.genfromtxt(GENDER_AGE_FILE_PATH, delimiter=',', skip_header=1)
    g_a_dict = {}
    for id, gender, age in zip(g_a_data[:, 0], g_a_data[:, 1], g_a_data[:, 2]):
        g_a_dict[int(id)] = (int(gender), int(age))
    return g_a_dict


def get_total_file_count():
    c = 0
    for dir in RAW_DATA_PATHS:
        c += len(os.listdir(dir))
    return c


def partition_train_test():
    user_id_list = np.genfromtxt(GENDER_AGE_FILE_PATH, delimiter=',', skip_header=1)[:, 0]
    partition = int(user_id_list.shape[0] * 0.7)  # Train:Test = 70:30
    train_user_id_list = user_id_list[:partition]
    test_user_id_list = user_id_list[partition:]
    return train_user_id_list, test_user_id_list


def append_other_columns(features, user_id, gender, age):
    user_id_column, gender_column, age_column = np.empty((features.shape[0], 1)), \
                                                np.empty((features.shape[0], 1)), \
                                                np.empty((features.shape[0], 1))
    user_id_column.fill(int(user_id))
    gender_column.fill(gender)
    age_column.fill(age)
    features = np.hstack((user_id_column, features))
    features = np.hstack((features, gender_column))
    features = np.hstack((features, age_column))
    return features


def send_to_partition(features, user_id, train_user_id_list, test_user_id_list, train_features, test_features):
    if user_id in train_user_id_list:
        if train_features is None:
            train_features = features
        else:
            train_features = np.vstack((train_features, features))
    elif user_id in test_user_id_list:
        if test_features is None:
            test_features = features
        else:
            test_features = np.vstack((test_features, features))
    return train_features, test_features


if __name__ == '__main__':

    train_user_id_list, test_user_id_list = partition_train_test()
    final_features, train_features, test_features = None, None, None
    GENDER_AGE_DICT = load_gender_age()
    total_file_count = get_total_file_count()
    file_counter, skip_counter = 0, 0

    for dir in RAW_DATA_PATHS:
        for filename in os.listdir(dir):

            filepath = dir + os.sep + filename
            user_id = int(filename.split('_')[1].replace('ID', ''))  # Filename: T0_ID000104_Center_seq0.csv

            print('User ID: ', user_id, end=' ')

            if user_id in GENDER_AGE_DICT:
                gender, age = GENDER_AGE_DICT[user_id]
            else:
                skip_counter += 1
                continue

            data = np.genfromtxt(filepath, delimiter=',', skip_header=2)  # Reading dataset
            features = get_features(data)  # Generating Features
            features = append_other_columns(features, user_id, gender, age)  # Append columns: user_id, gender, age
            train_features, test_features = send_to_partition(features, user_id, train_user_id_list,
                                                              test_user_id_list, train_features, test_features)

            # if final_features is None:
            #     final_features = features
            # else:
            #     final_features = np.vstack((final_features, features))

            file_counter += 1
            print('File #', file_counter + skip_counter, '/', total_file_count)

    # print('Shape of Final Feature Set: ', final_features.shape)
    print('Skipped Files: ', skip_counter)
    # np.save(SAVEPATH + os.sep + 'all_in_one_features.npy', final_features)
    # np.savetxt(SAVEPATH + os.sep + 'all_in_one_features.csv', final_features, delimiter=',')
    np.save(SAVEPATH + os.sep + 'train_all_in_one_features_' + str(WINDOW_LENGTH) + '.npy', train_features)
    np.save(SAVEPATH + os.sep + 'test_all_in_one_features_' + str(WINDOW_LENGTH) + '.npy', test_features)
