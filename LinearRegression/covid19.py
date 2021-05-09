
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def main():
    """
    main function that runs the program, building a suited linear regression model for the covid19_israel data
    described in the given csv file, analyzing the rate of infection using graph plotting and executing the missions
    described in the exercise.
    :return: nothing
    """
    covid19_design_matrix_T = pd.read_csv("covid19_israel.csv")
    log_detected = []
    for i in covid19_design_matrix_T.index:
        log_detected.append(math.log(covid19_design_matrix_T['detected'][i], 10))
    covid19_design_matrix_T["log_detected"] = log_detected

    y_response_vec = covid19_design_matrix_T["log_detected"]
    day_num_design_matrix_T = pd.DataFrame(covid19_design_matrix_T["day_num"])
    day_num_design_matrix_T.insert(0, "ones for bias", 1, True)
    day_num_design_matrix = day_num_design_matrix_T.transpose()  # according to exercise description
    # fit_linear_regression function should get the design matrix as argument (X) and not it's transpose (X_T)
    w_coefficients_vec = fit_linear_regression(day_num_design_matrix, y_response_vec)

    # plotting a graph describing the expected log of number of people detected according to the linear regression
    # model and the actual results
    log_detected_prediction_vec = predict(day_num_design_matrix, w_coefficients_vec)
    plt.scatter(covid19_design_matrix_T["day_num"], covid19_design_matrix_T["log_detected"], label="actual")
    plt.scatter(covid19_design_matrix_T["day_num"], log_detected_prediction_vec,  label="predicted")
    plt.xlabel("day_num")
    plt.ylabel("log_detected")
    plt.title("'log_detected' as a function of 'day_num'")
    plt.legend()
    plt.show()

    # plotting a graph describing the expected number of people detected according to the linear regression
    # model and the actual results
    detected_prediction_vec = 10 ** log_detected_prediction_vec
    plt.scatter(covid19_design_matrix_T["day_num"], covid19_design_matrix_T["detected"], label="actual")
    plt.scatter(covid19_design_matrix_T["day_num"], detected_prediction_vec, label="predicted")
    plt.xlabel("day_num")
    plt.ylabel("detected")
    plt.title("'detected' as a function of 'day_num'")
    plt.legend()
    plt.show()


def fit_linear_regression(X, y):
    """
    function execute linear regression on given design matrix (samples matrix) X and response vector y
    :param X: design matrix for linear regression execution
    :param y: response for linear regression execution
    :return: w coefficients vector got form a linear regression on X
    """
    X_dagger = np.linalg.pinv(X)
    w = np.dot(X_dagger.transpose(), y)
    return w


def predict(X, w):
    """
    function gets a design matrix X and coefficients vector w as arguments and returns prediction vector got by
    multiplication of X and w
    :param X: design matrix (samples matrix)
    :param w: coefficients vector
    :return: prediction vector got by multiplication of X and w
    """
    prediction_vector = np.dot(X.transpose(), w)
    return prediction_vector


if __name__ == "__main__":
    main()