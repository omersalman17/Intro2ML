
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    """
    main function that runs the program, builds a linear regression model, trains it and tests it over a real-world
    dataset of house prices and executes the missions described in the exercise.
    :return: nothing
    """
    kc_house_df_T, prices_vector = load_data("kc_house_data.csv")
    kc_house_df_T.insert(0, "one for bias", 1, True)  # inserting 1`s column for bias
    kc_house_df_design_matrix = kc_house_df_T.transpose()  # according to exercise description fit_linear_regression
    # should get the design matrix as argument (X) and not it's transpose (X_T)
    w, singular_values_matrix = fit_linear_regression(kc_house_df_design_matrix, prices_vector)
    plot_singular_values(singular_values_matrix)
    train_and_test_model(kc_house_df_T, prices_vector)
    feature_eval_df_design_matrix, y_prices_vector = get_design_matrix_and_response_vector_for_feature_evaluation()
    feature_evaluation(feature_eval_df_design_matrix, y_prices_vector)


def fit_linear_regression(X, y):
    """
    function execute linear regression on given design matrix (samples matrix) X and response vector y
    :param X: design matrix for linear regression execution
    :param y: response for linear regression execution
    :return: w coefficients vector got form a linear regression on X and y, S singular values matrix of design matrix X
    """
    S = np.linalg.svd(X, compute_uv=False)
    X_dagger = np.linalg.pinv(X)
    w = np.dot(X_dagger.transpose(), y)
    return w, S


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


def mse(response_vec, prediction_vec):
    """
    function gets a response vector and a prediction vector and returns their MSE
    :param response_vec: response vector
    :param prediction_vec: prediction vector
    :return: the MSE of the response vector and the prediction vector
    """
    response_vec = np.array(response_vec)
    prediction_vec = np.array(prediction_vec)
    mse = 0
    for i in range(len(response_vec)):
        mse += (prediction_vec[i] - response_vec[i])**2
    mse = mse / len(response_vec)
    return mse


def load_data(csv_file_path):
    """
    function loads the data from the csv file into Pandas DataFrame object, filters unsuited features and
    execute suitability actions  in order to make the data suited for building a linear regression model
    :param csv_file_path:
    :return: transpose of design matrix (samples matrix) and response vector suited for building a linear
    regression model from.
    """
    kc_house_df_T = pd.read_csv(csv_file_path)
    kc_house_df_T = kc_house_df_T.drop(columns="id")  # removing irrelevant 'id' column
    kc_house_df_T = kc_house_df_T.drop(columns="date")  # removing irrelevant 'date' column
    kc_house_df_T = kc_house_df_T.drop(columns="lat")  # removing irrelevant 'lat' column
    kc_house_df_T = kc_house_df_T.drop(columns="long")  # removing irrelevant 'long' column
    for col in kc_house_df_T:
        kc_house_df_T[col] = pd.to_numeric(kc_house_df_T[col], errors="coerce")  # changing values to numbers
        kc_house_df_T = kc_house_df_T.loc[kc_house_df_T[col] >= 0]  # checking validity of non-negative values
    kc_house_df_T = kc_house_df_T.dropna()  # removing rows containing empty data cells
    kc_house_df_T = pd.get_dummies(kc_house_df_T, columns=["zipcode"], drop_first=True)
    # transforming 'zipcode' column to dummies in order to use it properly in the model
    prices_vector = kc_house_df_T['price']
    kc_house_df_T = kc_house_df_T.drop(columns="price")  # removing response "price" vector
    return kc_house_df_T, prices_vector


def plot_singular_values(singular_values_collection):
    """
    function gets collection of singular values and plots them
    :param singular_values_collection:
    :return: nothing
    """
    singular_values_nparray = np.array(singular_values_collection)
    singular_values_nparray.sort()
    descending_order_sing_val_nparray = singular_values_nparray[::-1]
    plt.scatter(range(len(descending_order_sing_val_nparray)), descending_order_sing_val_nparray)
    plt.xlabel("index")
    plt.ylabel("singular value")
    plt.suptitle("singular values")
    plt.show()


def train_and_test_model(X_T, y):
    """
    function gets transpose of design matrix (X_T) and a response vector (y), splits the data into train and test sets,
    fitting a model to percentage of the training set and tests it`s performance using the tests set as described in
    the exercise - computes the MSEs and plot a suited graph according to each percentage of the training data.
    :param X_T: transpose of the design matrix
    :param y: response vector
    :return: nothing
    """
    X_T_train, X_T_test, y_train, y_test = train_test_split(X_T, y)  # default test_ and train_ size are 0.25 and 0.75
    X_test = X_T_test.transpose()
    mse_arr = []
    for i in range(1, 101):
        dividing_percentage = i / 100
        partial_X_T_train = X_T_train[:int(dividing_percentage * len(X_T_train.index))]
        partial_y_train = y_train[:int(dividing_percentage * len(y_train.index))]
        partial_X_train = partial_X_T_train.transpose()  # according to exercise description fit_linear_regression
        # should get the design matrix as argument (X) and not it's transpose (X_T)
        partial_w, partial_sing_val_matrix = fit_linear_regression(partial_X_train, partial_y_train)
        partial_prediction_vector = predict(X_test, partial_w)
        mse_arr.append(mse(y_test, partial_prediction_vector))
    plt.scatter(range(1, 101), mse_arr)
    plt.xlabel("percentage of training data samples")
    plt.ylabel("mse")
    plt.suptitle("mse on percentage of the training data samples")
    plt.show()


def feature_evaluation(X, y_response_vector):
    """
    function gets a design matrix (X) and a response vector (y), plots a different graph for each feature`s values and
    the response vector`s values and describes their Pearson Correlation as described in the exercise, on that way
    according to the graphs ploted we can get a clue about what feature are more relevant for estimating the response
    vector - the prices vector so we could build the linear regression model to be more accurate
    :param X: design matrix (samples matrix)
    :param y_response_vector: response vector
    :return: nothing
    """
    X_T = X.transpose()  # according to exercise description feature_evaluation function should get the design matrix
    # as argument (X) and not it's transpose (X_T)
    y_response_vector_std = np.std(y_response_vector)
    for col in X_T:
        col_values_vector = X_T[col]
        plt.scatter(col_values_vector, y_response_vector)
        plt.xlabel(col)
        plt.ylabel('price')
        vectors_cov = np.cov(col_values_vector, y_response_vector)
        col_values_vector_std = np.std(col_values_vector)
        Pearson_Correlation_m = vectors_cov / (col_values_vector_std * y_response_vector_std)
        Pearson_Corr = Pearson_Correlation_m[0][1] / np.sqrt(Pearson_Correlation_m[0][0] * Pearson_Correlation_m[1][1])
        plt.suptitle("house " + col + " and it`s price\n\n Pearson Correlation: " + str(Pearson_Corr))
        plt.show()


def get_design_matrix_and_response_vector_for_feature_evaluation():
    """
    function reads the data into a DataFrame object, filters it and returns transpose of design matrix and
    response vector suited for feature evaluation.
    :return: transpose of design matrix and response vector suited for feature evaluation.
    """
    kc_house_df_T = pd.read_csv("kc_house_data.csv")
    kc_house_df_T.drop(columns="zipcode")  # filtering categorical features
    kc_house_df_T['date'] = pd.to_datetime(kc_house_df_T['date'], errors="coerce").dt.strftime("%y%m%d")
    # changing date feature to suited format
    for col in kc_house_df_T:  # filtering illegal data
        kc_house_df_T[col] = pd.to_numeric(kc_house_df_T[col], errors="coerce")  # changing values to numbers
        if col != "long":
            kc_house_df_T = kc_house_df_T.loc[kc_house_df_T[col] >= 0]  # checking validity of non-negative values
    kc_house_df_T = kc_house_df_T.dropna()  # removing rows containing empty data cells
    y_prices_vector = kc_house_df_T['price']
    kc_house_df_T = kc_house_df_T.drop(columns="price")  # removing response "price" vector
    return kc_house_df_T.transpose(), y_prices_vector


if __name__ == "__main__":
    main()