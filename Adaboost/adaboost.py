"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex_4.ex4_tools import *
import matplotlib.pyplot as plt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        # sample size
        m = len(y)
        # initialize weights (for 0's iteration) uniformly
        D_t = np.ones(shape=m) / m
        for t in range(self.T):
            # fit  weak learner
            self.h[t] = self.WL(D_t, X, y)
            # calculate error and weights from weak learner prediction
            prediction_t = self.h[t].predict(X)
            epsilon_t = D_t[(prediction_t != y)].sum()
            self.w[t] = np.log((1 - epsilon_t) / epsilon_t) / 2
            # update sample weights
            D_t = D_t * np.exp(-self.w[t] * y * prediction_t)
            D_t /= D_t.sum()
        return D_t

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        prediction = np.zeros(shape=(len(X)))
        if max_t > self.T:
            max_t = self.T
        for i in range(max_t):
            prediction += self.w[i] * (self.h[i].predict(X))
        return np.sign(prediction)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        prediction = self.predict(X, max_t)
        correct_predictions = prediction[(prediction != y)].sum()
        return correct_predictions / len(y)


def question10(AdaBoost_clf, X_train, y_train, X_test, y_test, test_errors_ratio_arr, noise):
    """
    function answers question 10 demands described in the exercise, computes and plots the demanded graph.
    :param AdaBoost_clf: Adaboost object contains a weak learner and other needed variables for Adaboost properties.
    :param X_train: training samples.
    :param y_train: true labeling of training samples.
    :param X_test: test samples.
    :param y_test: labeling of test samples (received from our classifier).
    :param test_errors_ratio_arr: array of test errors ratios according to number of iterations (T) of Adaboost.
    :param noise: noise.
    :return: nothing.
    """
    train_errors_ratio_arr = []
    T_array = np.arange(1, 501)
    for t in T_array:
        train_errors_ratio_arr.append(AdaBoost_clf.error(X_train, y_train, t))
        test_errors_ratio_arr.append(AdaBoost_clf.error(X_test, y_test, t))
    plt.plot(T_array, train_errors_ratio_arr, color="blue", label="train samples")
    plt.plot(T_array, test_errors_ratio_arr, color="orange", label="test samples")
    plt.xlabel("number of iterations for Adaboost (T)")
    plt.ylabel("Error ratio")
    plt.title("Error ratio as a function of T \n noise: " + str(noise))
    plt.legend(title="samples")
    plt.show()


def question11(AdaBoost_clf, X_test, y_test, noise):
    """
    function answers question 11 demands described in the exercise, computes and plots the demanded graph.
    :param AdaBoost_clf: Adaboost object contains a weak learner and other needed variables for Adaboost properties.
    :param X_test: test samples.
    :param y_test: labeling of test samples (received from our classifier).
    :param noise: noise.
    :return: nothing.
    """
    T_array = [5, 10, 50, 100, 200, 500]
    i = 1
    for T in T_array:
        plt.subplot(2, 3, i, title="T is " + str(T) + "\nnoise: " + str(noise))
        decision_boundaries(AdaBoost_clf, X_test, y_test, num_classifiers=T)
        i += 1
    plt.show()


def question12(AdaBoost_clf, X_train, y_train, test_errors_ratio_arr, noise):
    """
    function answers question 12 demands described in the exercise, computes and plots the demanded graph.
    :param AdaBoost_clf: Adaboost object contains a weak learner and other needed variables for Adaboost properties.
    :param X_train: training samples.
    :param y_train: true labeling of training samples.
    :param test_errors_ratio_arr: array of test errors ratios according to number of iterations (T) of Adaboost.
    :param noise: noise.
    :return: nothing.
    """
    minimal_T = np.argmin(test_errors_ratio_arr)
    decision_boundaries(AdaBoost_clf, X_train, y_train, num_classifiers=minimal_T)
    plt.title("decision boundaries of Adaboost on training data when minimal T  = " + str(minimal_T) +
              "\nnoise: " + str(noise) + "\ntest_error_ratio: " + str(test_errors_ratio_arr[int(minimal_T)]))
    plt.show()


def question13(AdaBoost_clf, X_train, y_train, noise, D_t):
    """
    function answers question 13 demands described in the exercise, computes and plots the demanded graph.
    :param AdaBoost_clf: Adaboost object contains a weak learner and other needed variables for Adaboost properties.
    :param X_train: training samples.
    :param y_train: true labeling of training samples.
    :param noise: noise.
    :param D_t: weights of the samples used for trainings the model in the last iteration (T = 500).
    :return: nothing.
    """
    normalized_weights = (D_t / np.max(D_t)) * 10
    decision_boundaries(AdaBoost_clf, X_train, y_train, num_classifiers=AdaBoost_clf.T, weights=normalized_weights)
    plt.title("decision boundaries of Adaboost on training data when T = 500" + "\nnoise: " + str(noise))
    plt.show()


def main():
    """
    main function the execute the program, contains implementation for question 14 as described in the exercise.
    :return: nothing
    """
    number_of_iterations_for_adaboost = 500
    number_of_train_samples_to_generate = 5000
    number_of_test_samples_to_generate = 200
    noise_arr = [0, 0.01, 0.4]
    AdaBoost_clf = AdaBoost(DecisionStump, number_of_iterations_for_adaboost)
    for noise in noise_arr:
        X_train, y_train = generate_data(number_of_train_samples_to_generate, noise)
        D_t = AdaBoost_clf.train(X_train, y_train)
        X_test, y_test = generate_data(number_of_test_samples_to_generate, noise)
        test_errors_ratio_arr = []
        question10(AdaBoost_clf, X_train, y_train, X_test, y_test, test_errors_ratio_arr, noise)
        question11(AdaBoost_clf, X_test, y_test, noise)
        question12(AdaBoost_clf, X_train, y_train, test_errors_ratio_arr, noise)
        question13(AdaBoost_clf, X_train, y_train, noise, D_t)


if __name__ == "__main__":
    main()