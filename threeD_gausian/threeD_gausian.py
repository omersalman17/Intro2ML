import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    
def main():
    #%%
    # Q_11

    # from threeD_gausian import * // needed when using the code in an external jupyterLab file
    plot_3d(x_y_z)
    plt.suptitle("plot of 50000 3D samples")

    #%%
    # Q_12

    scalingMatrix = [[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]]
    scaledMatrix = np.dot(scalingMatrix, x_y_z)
    plot_3d(scaledMatrix)
    print("covariance matrix is:")
    print(np.cov(scaledMatrix))
    plt.suptitle("plot of the samples matrix multiplied by scaling matrix")

    #%%
    # Q_13

    orthogonal_matrix = get_orthogonal_matrix(3)
    newMatrix = np.dot(orthogonal_matrix, scaledMatrix)
    plot_3d(newMatrix)
    print("covariance matrix is:")
    print(np.cov(newMatrix))
    plt.suptitle("plot of the scaled matrix multiplied by random orthogonal matrix")

    #%%
    # Q_14

    plot_2d(x_y_z)
    plt.suptitle("plot of the prijection of the data (samples matrix) to the x, y, axes")

    #%%
    # Q_15

    not_suited_vectors_indexes = []
    for i in range(len(x_y_z[2])):
        if (x_y_z[2][i] <= -0.4 or x_y_z[2][i] >= 0.1):
            not_suited_vectors_indexes.append(i)
    suitedMatrix = np.delete(x_y_z, not_suited_vectors_indexes, axis=1)
    plot_2d(suitedMatrix)
    plt.suptitle("plot of the projection of the points whose  0.1 > z > -0.4 to the x,y axes")

    #%%
    # Q_16_A

    data = np.random.binomial(1, 0.25, (100000, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    plt.xlabel("m")
    plt.ylabel("average")
    for i in range(5):
        avg_vector = np.cumsum(data[i]) / np.arange(1, 1001);
        plt.plot(avg_vector, label="sample number " + str(i))
        plt.legend()
    plt.suptitle("plot of the average estimator for the first 5 samples as a function of m")

    #%%
    # Q_16_B+C

    for e in epsilon:
        chebeyshev_bounds_list = []
        hoeffding_bounds_list = []
        for m in range(1, 1001):
            chebeyshev_bound = 1 / (4 * m * e ** 2)
            chebeyshev_bounds_list.append(min(1, chebeyshev_bound))
            hoeffding_bound = 2 * np.e ** (-2 * m * e ** 2)
            hoeffding_bounds_list.append(min(1, hoeffding_bound))
        sat_seq_indicators_list = []  # sat_seq = satisfying_sequences
        for row in data:
            row_sum_by_m = row.cumsum() / np.arange(1, 1001)
            sat_seq_indicators = np.abs(row_sum_by_m - 0.25) >= e
            sat_seq_indicators_list.append(sat_seq_indicators)
        sat_seq_precentages = np.sum(np.array(sat_seq_indicators_list).T, axis=1) / 100000
        plt.xlabel("m")
        plt.ylabel("expression satisfying probability")
        plt.plot(sat_seq_precentages, label="satisfying sequences precentage")
        plt.suptitle("plot of the probability to satisfy the expression as a function of m when epsilon is: " + str(e))
        plt.plot(chebeyshev_bounds_list, label="chebeyshev bound")
        plt.plot(hoeffding_bounds_list, label="hoeffding bound")
        plt.legend()
        plt.show()

    #%%