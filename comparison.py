import numpy as np
import matplotlib.pyplot as plt
import models


def draw_points(m):
    """
    this function gets an integer m and returns a pair X, y where X is 2 x m matrix where each column represents an
    i.i.d sample from the a two dimensional Gaussian with mean vector of zeros and a unit matrix for covariance
    distribution and y is its corresponding label, according to to f function described in the exercise.
    :param m: given int
    :return: a pair X, y as described above
    """
    mean_vector = [0, 0]
    cov_matrix = [[1, 0], [0, 1]]
    X_T = np.random.multivariate_normal(mean_vector, cov_matrix, m)
    y = np.sign(np.dot([0.3, -0.5], X_T.transpose()) + 0.1)   # todo: syntax
    return X_T.transpose(), y


def question9():
    """
    function answers question 9 demands described in the exercise, computes and plots the demanded graphs.
    :return: nothing
    """
    m_array = [5, 10, 15, 25, 70]
    for i in m_array:
        X, y_train = draw_points(i)
        X_T = X.transpose()
        P_xi_lst = []  # list of all samples classified y_train = 1
        N_xi_lst = []  # list of all samples classified y_train = -1
        for j in range(len(y_train)):
            if y_train[j] == 1:
                P_xi_lst.append(X_T[j])
            else:
                N_xi_lst.append(X_T[j])
        P_xi_lst = np.array(P_xi_lst)
        N_xi_lst = np.array(N_xi_lst)
        P_xi_lst_T = P_xi_lst.transpose()
        N_xi_lst_T = N_xi_lst.transpose()
        plt.scatter(x=P_xi_lst_T[0], y=P_xi_lst_T[1], label='classified positive')
        plt.scatter(x=N_xi_lst_T[0], y=N_xi_lst_T[1], label='classified negative')
        plt.xlabel("first coordinate value")
        plt.ylabel("second coordinate value")
        plt.title(str(i) + " samples classification")
        w = [0.3, -0.5]
        intercept = 0.1
        a = -w[0] / w[1]
        xx = np.linspace(np.min(X), np.max(X))
        yy = a * xx + (-intercept) / w[1]
        plt.plot(xx, yy, color="green", label='given f hyperplane')
        perceptron = models.Perceptron()
        perceptron.fit(X, y_train)
        w = perceptron.model
        a = -w[1] / w[2]
        yy = a * xx + -w[0] / w[2]
        plt.plot(xx, yy, color="red", label='perceptron hyperplane')
        svm = models.SVM()
        svm.fit(X, y_train)
        w = svm.model.coef_[0]
        a = -w[0] / w[1]
        yy = a * xx - (svm.model.intercept_[0]) / w[1]
        plt.plot(xx, yy, color="purple", label='SVM hyperplane')
        plt.legend()
        plt.show()


def question10():
    """
    function answers question 10 demands described in the exercise, computes and plots the demanded graph.
    :return: nothing
    """
    m_array = [5, 10, 15, 25, 70]
    Perceptron_ma_lst, SVM_ma_lst, LDA_ma_lst = [], [], []  # ma = mean accuracy
    for i in m_array:
        Perceptron_acc_sum, SVM_acc_sum, LDA_acc_sum = 0, 0, 0   # ma = mean accuracy
        for j in range(500):
            X_train, y_train = draw_points(i)
            if -1 not in y_train or 1 not in y_train:
                j -= 1
                continue
            X_test, y_test = draw_points(10000)
            Perceptron_clf = models.Perceptron()
            Perceptron_clf.fit(X_train, y_train)
            Perceptron_pd = Perceptron_clf.score(X_test, y_test)  # pd = performance dictionary
            Perceptron_acc_sum += Perceptron_pd["accuracy"]
            SVM_clf = models.SVM()
            SVM_clf.fit(X_train, y_train)
            SVM_pd = SVM_clf.score(X_test, y_test)  # pd = performance dictionary
            SVM_acc_sum += SVM_pd["accuracy"]
            LDA_clf = models.LDA()
            LDA_clf.fit(X_train, y_train)
            LDA_pd = LDA_clf.score(X_test, y_test)  # pd = performance dictionary
            LDA_acc_sum += LDA_pd["accuracy"]
        Perceptron_ma_lst.append(Perceptron_acc_sum/500)
        SVM_ma_lst.append(SVM_acc_sum/500)
        LDA_ma_lst.append(LDA_acc_sum/500)
    plt.plot(m_array, Perceptron_ma_lst, color="blue", label="Percepton")
    plt.plot(m_array, SVM_ma_lst, color="red", label="SVM")
    plt.plot(m_array,LDA_ma_lst, color="green", label="LDA")
    plt.xlabel("number of samples")
    plt.ylabel("classifier mean accuracy")
    plt.title("classifiers performance")
    plt.legend(title="classifiers")
    plt.show()


if __name__ == "__main__":
    question9()
    question10()
