import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class MLmodel:

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        pass

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        pass

    def score(self, X, y):
        """
         Given an unlabeled test set X and the true labels y of this test set, returns a dictionary with the following
        fields: • num samples: number of samples in the test set
                • error: error (misclassification) rate
                • accuracy: accuracy
                • FPR: false positive rate
                • TPR: true positive rate
                • precision: precision
                • recall: recall
        :param X: unlabeled test set
        :param y: the true labels of this test set
        :return: dictionary contains the performance of the classifier
        """
        performance_dictionary = {}
        y_hat = self.predict(X)  # y_hat = predicted classification vector
        TP_num = np.sum(np.logical_and(y == 1, y_hat == 1))
        FP_num = np.sum(np.logical_and(y == -1, y_hat == 1))
        TN_num = np.sum(np.logical_and(y == -1, y_hat == -1))
        FN_num = np.sum(np.logical_and(y == 1, y_hat == -1))
        P_num = np.sum(y == 1)
        N_num = len(y) - P_num
        num_samples = len(X)
        performance_dictionary["num_samples"] = num_samples
        performance_dictionary["error"] = (FP_num + FN_num) / (P_num + N_num)
        performance_dictionary["accuracy"] = (TP_num + TN_num) / (P_num + N_num)
        performance_dictionary["FPR"] = FP_num / N_num
        performance_dictionary["TPR"] = TP_num / P_num
        performance_dictionary["precision"] = TP_num / (TP_num + FP_num)
        performance_dictionary["recall"] = TP_num / P_num
        return performance_dictionary


class Perceptron(MLmodel):

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        X = np.insert(X, 0, 1, axis=0)
        X_T = X.transpose()  # working with X.transpose() for comfortable indexing
        X = np.array(X_T.transpose())
        features_num, samples_num = np.array(X).shape
        w = np.zeros(len(X), dtype=np.int16)  # len(X) = number of rows in X = number of features of each sample on X
        y = np.array(y)
        while True:
            found = False
            for i in range(samples_num):
                if y[i] * np.dot(w, X_T[i]) <= 0:
                    w = w + y[i] * X_T[i]
                    found = True
                    break
            if not found:
                self.model = w
                break

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        X = np.insert(X, 0, 1, axis=0)
        X_T = X.transpose()
        y_hat = np.dot(X_T, self.model)  # y_hat = predicted classification vector
        for i in range(len(y_hat)):
            if y_hat[i] > 0:
                y_hat[i] = 1
            elif y_hat[i] < 0:
                y_hat[i] = -1
        return y_hat


class LDA(MLmodel):

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        samples_num = len(y)
        P_num = np.sum(y == 1)
        N_num = samples_num - P_num
        X_T = X.transpose()
        y_P_xi_sum = np.sum(X_T[y == 1], axis=0)  # sum of vectors xi who`s yi = 1  # todo: syntax
        y_N_xi_sum = np.sum(X_T[y == -1], axis=0)  # sum of vectors xi who`s yi = -1
        y_P_mean_est = y_P_xi_sum / P_num  # y = 1 (Positive) mean estimator
        y_N_mean_est = y_N_xi_sum / N_num  # y = -1 (Negative) mean estimator
        cov_matrix_inverse = np.linalg.inv(np.cov(X))
        y_P_prob_est = P_num / samples_num  # P(y = 1) estimator
        y_N_prob_est = N_num / samples_num  # P(y = -1) estimator
        self.model = [y_P_mean_est, y_N_mean_est, cov_matrix_inverse, y_P_prob_est, y_N_prob_est]

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        X_T = X.transpose()
        y_P_mean_est = self.model[0]
        y_N_mean_est = self.model[1]
        cov_matrix_inverse = self.model[2]
        y_P_prob_est = self.model[3]
        y_N_prob_est = self.model[4]
        delta_y_P_vec = np.dot(np.dot(X_T, cov_matrix_inverse), y_P_mean_est) - 0.5 * \
                    np.dot(np.dot(y_P_mean_est.transpose(), cov_matrix_inverse), y_P_mean_est) + \
                    np.log(y_P_prob_est)
        # delta_y=1_(X)
        delta_y_N_vec = np.dot(np.dot(X_T, cov_matrix_inverse), y_N_mean_est) - 0.5 * \
                    np.dot(np.dot(y_N_mean_est.transpose(), cov_matrix_inverse), y_N_mean_est) + \
                    np.log(y_N_prob_est)
        # delta_y=-1_(X)
        y_hat = np.where(delta_y_N_vec < delta_y_P_vec, 1, -1)  # y_hat= predicted classification vector
        return y_hat


class SVM(MLmodel):

    def __init__(self):
        super().__init__()
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        self.model.fit(X.transpose(), y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        return self.model.predict(X.transpose())


class Logistic(MLmodel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        return self.model.predict(X)


class DecisionTree(MLmodel):
    def __init__(self, max_depth):
        super().__init__()
        self.DecisionTree_obj = DecisionTreeClassifier(max_depth)

    def fit(self, X, y):
        """
        Given a training set as X and y this method learns the parameters of the model and stores the trained model
        (namely, the variables that define hypothesis chosen) in self.model. The method returns nothing.
        :param X: design matrix so that each column is a sample
        :param y: response / labels vector
        :return: nothing
        """
        self.model = self.DecisionTree_obj.fit(X, y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels
        :param X: unlabeled test set X
        :return: vector of predicted labels
        """
        return self.DecisionTree_obj.predict(X)
