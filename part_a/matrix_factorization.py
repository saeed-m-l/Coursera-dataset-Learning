from utils import *
from scipy.linalg import sqrtm

import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def predict_nan_values(matrix, n_components):
    # nan_indices = np.argwhere(np.isnan(matrix))
    # filled_matrix = matrix.copy()
    # filled_matrix[np.isnan(filled_matrix)] = 0
    #
    # U, s, Vt = np.linalg.svd(filled_matrix)
    # filled_matrix = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    #
    # for i, j in nan_indices:
    #     matrix[i, j] = filled_matrix[i, j]

    # return matrix
    return 4


# need to be clear
def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.

    # #####################################################################
    # # TODO:                                                             #
    # # Part A:                                                           #
    # # Implement the function as described in the docstring.             #
    # #####################################################################
    # dense_matrix = matrix.toarray()
    # filled_matrix = np.nan_to_num(dense_matrix)
    # filled_matrix = csr_matrix(filled_matrix)
    # U, S, Vt = svds(filled_matrix, k=k)
    # reconst_matrix = np.dot(U, np.dot(np.diag(S), Vt))
    # # choose best K
    # #####################################################################
    # #                       END OF YOUR CODE                            #
    # #####################################################################
    # return np.array(reconst_matrix)
    matrix_filled = np.nan_to_num(matrix)

    U, S, Vt = svd(matrix_filled, full_matrices=False)
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]
    filled_matrix = np.copy(matrix)
    # Find the indices of NaN cells
    row_indices, col_indices = np.where(np.isnan(matrix))
    # Fill in the missing values
    for row, col in zip(row_indices, col_indices):
        filled_matrix[row, col] = np.dot(U[row, :], np.dot(np.diag(S), Vt[:, col]))
        if filled_matrix[row, col] >= 0.5:
            filled_matrix[row, col] = 1
        else:
            filled_matrix[row, col] = 0

    return filled_matrix


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Update the corresponding elements of U and Z
    u[n] += lr * (c - np.dot(u[n], z[q])) * z[q]
    z[q] += lr * (c - np.dot(u[n], z[q])) * u[n]

    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    # u = np.random.uniform(low=0, high=1 / np.sqrt(k),
    #                       size=(len(set(train_data["user_id"])), k))
    # z = np.random.uniform(low=0, high=1 / np.sqrt(k),
    #                       size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Initialize U and Z
    num_users = len(set(train_data["user_id"]))
    num_questions = len(set(train_data["question_id"]))
    u = np.random.uniform(low=0, high=1 / np.sqrt(k), size=(num_users, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k), size=(num_questions, k))

    # Perform ALS iterations
    for _ in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)

    # Reconstruct the matrix using U and Z
    mat = u @ z.T
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] >= 0.5:
                mat[i][j] = 1
            else:
                mat[i][j] = 0
    return mat
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def matrix_factorization_main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # part 1
    K = [5, 10, 20, 50, 100]
    best_acc_svd = 0
    for k in K:
        acc = sparse_matrix_evaluate(val_data, svd_reconstruct(train_matrix, k))
        print('The accuracy in SVD for k=%d is %f' % (
            k, acc))
        if acc >= best_acc_svd:
            best_val_acc_svd = acc
            best_k_svd = k
    for k in K:
        acc = sparse_matrix_evaluate(val_data, als(train_data, k, 0.08, 80000))
        print('The accuracy in ALS for k=%d is %f' % (
            k, acc))
        if acc >= best_acc_svd:
            best_val_acc_als = acc
            best_k_als = k
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Part D and E:                                                     #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # Results of part D
    # Results of part E
    # Save the line chart
    import matplotlib.pyplot as plt
    itr = np.linspace(20000, 80000, 100)
    acc = []
    for i in range(len(itr)):
        acc.append(sparse_matrix_evaluate(val_data, als(train_data, best_k_als, 0.08, int(itr[i]))))
    plt.figure()
    plt.plot(itr, acc)
    plt.xlabel('Itr')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Vs Itr')
    #plt.savefig('/part_e.png')
    plt.savefig('../plots/matrix_factorization/MF.png')
    test_acc_svd = 0
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    results = {
        'best_k_svd': best_k_svd,
        'test_acc_svd': test_acc_svd,
        'best_val_acc_svd': best_val_acc_svd,
        'best_val_acc_als': best_val_acc_als,
        'best_k_als': best_k_als

    }

    return results


if __name__ == "__main__":
    matrix_factorization_main()
