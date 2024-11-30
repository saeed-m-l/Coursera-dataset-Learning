from sklearn.impute import KNNImputer
from utils import *
#####################################################################                                                        #
# Import packages you need here                                     #
#####################################################################
import numpy as np
import copy
import matplotlib.pyplot as plt


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    im_matrix = imputer.fit_transform(matrix)
    completed_mat = copy.deepcopy(matrix)
    completed_mat[np.isnan(matrix)] = im_matrix[np.isnan(matrix)]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################    
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    print("Validation Accuracy with k = %d is %f "%(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    imputer = KNNImputer(n_neighbors=k)
    im_matrix = imputer.fit_transform(matrix.T)
    completed_mat = copy.deepcopy(matrix.T)
    completed_mat[np.isnan(matrix.T)] = im_matrix[np.isnan(matrix.T)]
    acc = sparse_matrix_evaluate(valid_data, completed_mat.T)
    print("Validation Accuracy with k = %d is %f "%(k,acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    # try:
    #     test_data = load_public_test_csv("../data")
    # except NameError:
    #     print('We do not have test data')
    print(sparse_matrix.shape)
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Part B&C:                                                         #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_user().                                                       #
    #####################################################################
    user_best_k: float = 0  # :float means that this variable should be a float number
    user_test_acc: float = None
    user_valid_acc: list = []
    best_acc = 0
    K = np.array([1, 6, 11, 16, 21, 26])
    for k in K:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_valid_acc.append(acc)
        if acc >= best_acc:
            best_acc = acc
            user_best_k = k
    plt.figure()
    plt.plot(K, user_valid_acc)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN by user')
    plt.savefig('../plots/knn/users.png')
    print('Best K in KNN imputed by user is : %f and its accuracy is : %f' % (user_best_k, best_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_item().                                                        #
    #####################################################################
    pass
    question_best_k: float = 0  # :float means that this variable should be a float number
    question_test_acc: float = None
    question_valid_acc: list = []
    best_acc = 0
    for k in K:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        question_valid_acc.append(acc)
        if acc >= best_acc:
            best_acc = acc
            question_best_k = k
    plt.figure()
    plt.plot(K, question_valid_acc)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN by question')
    plt.savefig('../plots/knn/question.png')
    print('Best K in KNN imputed by user is : %f and its accuracy is : %f' % (question_best_k, best_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
        'user_best_k': user_best_k,
        'user_test_acc': user_test_acc,
        'user_valid_accs': user_valid_acc,
        'question_best_k': question_best_k,
        'question_test_acc': question_test_acc,
        'question_valid_acc': question_valid_acc,
    }

    return results


if __name__ == "__main__":
    knn_main()
