import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC
from sklearn import utils

DATAPATH = r'../features'
# DATAFILE = 'all_in_one_features.npy'
TRAIN_DATAFILE = 'train_all_in_one_features_200.npy'
TEST_DATAFILE = 'test_all_in_one_features_200.npy'
CV = 10  # K in K-fold Cross Validation

def predict_age():
    pass

def predict_gender(X_train, y_train, X_test, y_test):
    classifiers = {
        # 'Baseline Classifier': DummyClassifier(),
        'Random Forest Classifier': RandomForestClassifier(max_depth=15, n_estimators=20),
        'KNN': KNeighborsClassifier(10),
        # 'Decision Tree': DecisionTreeClassifier(max_depth=6),
        # 'Linear SVM': SVC(kernel="linear", C=0.05),
        # 'RBF SVM': SVC(gamma=2, C=1)
    }
    classifiers_title = list(classifiers.keys())
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    # Performing classification using each classifier and computing the 10-Fold cross-validation on results
    acc = 0
    for i in range(classifiers.__len__()):
        print('========================================================')
        print('Running:', classifiers_title[i])
        classifiers[classifiers_title[i]].fit(X_train, y_train)
        y_pred = classifiers[classifiers_title[i]].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy:', acc)
        print('========================================================')
    return acc


def load_dataset():
    train_dataset = np.load(DATAPATH + os.sep + TRAIN_DATAFILE)
    test_dataset = np.load(DATAPATH + os.sep + TEST_DATAFILE)
    X_train = np.nan_to_num(train_dataset[:, 1:-2])
    y_train = np.nan_to_num(train_dataset[:, -2])
    X_test = np.nan_to_num(test_dataset[:, 1:-2])
    y_test = np.nan_to_num(test_dataset[:, -2])
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_dataset()


    acc_score_gender = predict_gender(X_train, y_train, X_test, y_test)

    print(acc_score_gender)