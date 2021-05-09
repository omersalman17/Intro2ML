from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClf


def rearrange_data(x):
    return np.reshape(x, (np.size(x, 0), 784))


def question12():
    """
    function answers question 12 demands described in the exercise
    :return: train_data, x_test, y_test suited for the question demands described in the exercise
    """
    train_data = np.empty([60000, 785])
    row = 0
    for line in open("mnist_train.csv"):
        train_data[row] = np.fromstring(line, sep=",")
        row += 1
    test_data = np.empty([10000, 785])
    row = 0
    for line in open("mnist_test.csv"):
        test_data[row] = np.fromstring(line, sep=",")
        row += 1
    y_train = train_data[:, 0]
    x_train = train_data[:, 1:]
    y_test = test_data[:, 0]
    x_test = test_data[:, 1:]
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]
    x_train_reshape = np.reshape(x_train, (12665, 28, 28))
    ones = np.where(y_train == 1)[0]
    zeroes = np.where(y_train == 0)[0]
    for i in range(3):
        plt.imshow(x_train_reshape[ones[i]])
        plt.show()
        plt.imshow(x_train_reshape[zeroes[i]])
        plt.show()
    return train_data, x_test, y_test


def question_14():
    """
    function answers question 14 demands described in the exercise, computes and plots the demanded graphs.
    :return: nothing
    """
    train_data, X_test, y_test = question12()
    y_train = train_data[:, 0]
    ones_or_zeroes = np.logical_or(y_train == 0, y_train == 1)
    train_data = train_data[ones_or_zeroes]
    m_array = [50, 100, 300, 500]
    Log_Reg_ma_lst, SVM_ma_lst, DecisionTree_ma_lst, KNN_ma_lst = [], [], [], []  # ma = mean accuracy
    for m in m_array:
        Log_Reg_acc_sum, SVM_acc_sum, DecisionTree_acc_sum, KNN_acc_sum = 0, 0, 0, 0   # ma = mean accuracy
        for j in range(50):
            chosen_samples = random.sample(list(train_data), m)
            chosen_samples = np.array(chosen_samples)
            X_train = chosen_samples[:, 1:]
            y_train = chosen_samples[:, 0]
            if 0 not in y_train or 1 not in y_train:
                j -= 1
                continue
            Log_Reg_clf = LogisticRegression(solver='liblinear')
            Log_Reg_clf.fit(X_train, y_train)
            Log_Reg_acc_sum += Log_Reg_clf.score(X_test, y_test)
            SVM_clf = SVC(C=1e10, kernel='linear')
            SVM_clf.fit(X_train, y_train)
            SVM_acc_sum += SVM_clf.score(X_test, y_test)
            DecisionTree_clf = DecisionTreeClf(max_depth=10)
            DecisionTree_clf.fit(X_train, y_train)
            DecisionTree_acc_sum += DecisionTree_clf.score(X_test, y_test)
            KNN_clf = KNeighborsClassifier(n_neighbors=2)
            KNN_clf.fit(X_train, y_train)
            KNN_acc_sum += KNN_clf.score(X_test, y_test)
        Log_Reg_ma_lst.append(Log_Reg_acc_sum/50)
        SVM_ma_lst.append(SVM_acc_sum/50)
        DecisionTree_ma_lst.append(DecisionTree_acc_sum/50)
        KNN_ma_lst.append(KNN_acc_sum/50)
    plt.plot(m_array, Log_Reg_ma_lst, color="blue", label="Log_Reg")
    plt.plot(m_array, SVM_ma_lst, color="red", label="SVM")
    plt.plot(m_array, DecisionTree_ma_lst, color="green", label="DecisionTree")
    plt.plot(m_array, KNN_ma_lst, color="purple", label="KNN")
    plt.xlabel("number of samples")
    plt.ylabel("classifier mean accuracy")
    plt.title("classifiers performance")
    plt.legend(title="classifiers")
    plt.show()


if __name__ == "__main__":
    question_14()
