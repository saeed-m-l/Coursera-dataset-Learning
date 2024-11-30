import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
import copy
from utils import *


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
    return completed_mat


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
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return completed_mat.T


def get_intelligence(data, y):
    unique_students, student_indices = np.unique(data, return_inverse=True)
    num_students = len(unique_students)

    intelligence = np.zeros((num_students, 2))

    for i in range(num_students):
        student_id = unique_students[i]

        student_answers = y[data == student_id]
        intelligence[i, 0] = student_id
        intelligence[i, 1] = np.mean(student_answers)

    return intelligence


def multiply_and_threshold(array1, array2):
    multiplied = array1 * array2
    threshold = np.where(multiplied > 0.5, 1, 0)
    return threshold


def threshold_processing(array1, array2, array3):
    multiplied = array1 * array2
    multiplied2 = array1 * array3
    threshold = np.where(multiplied > 0.5, 1, 0)
    for i in range(len(array1)):
        if multiplied[i] >= 0.5 and multiplied2[i] >= 0.5:
            threshold[i] = 1
        elif multiplied[i] >= 0.5 or multiplied2[i] >= 0.5:
            if array1[i] >= 0.5:
                threshold[i] = 1
            else:
                threshold[i] = 0
        else:
            threshold[i] = 0
    return threshold


def map_intelligence_to_test_data(indices, matrix, train_intelligence, max=1.5, min=0.8):
    for i in range(len(indices[0])):
        train_indic = np.where(train_intelligence[:, 0] == indices[0][i])[0]
        if len(train_indic) == 1:
            train_int = train_intelligence[train_indic[0], 1]
            if train_int >= 0.5:
                tmp = max * matrix[indices[0][i], indices[1][i]]
                if tmp >= 0.5:
                    matrix[indices[0][i], indices[1][i]] = 1
                else:
                    matrix[indices[0][i], indices[1][i]] = 0
            else:
                tmp = min * matrix[indices[0][i], indices[1][i]]
                if tmp >= 0.5:
                    matrix[indices[0][i], indices[1][i]] = 1
                else:
                    matrix[indices[0][i], indices[1][i]] = 0


def enhancement_main():
    # I will add gender to it
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    subject = load_subject_csv("../data")
    train = load_train_csv("../data")
    complete_matrix = knn_impute_by_item(sparse_matrix, val_data, 21)
    missing_indices = np.where(np.isnan(sparse_matrix))
    y = np.array(train['is_correct'])
    X1 = np.array(train['question_id'])
    X2 = np.array(train['user_id'])
    student_intel = get_intelligence(X2, y)
    # Here add One Hot encoder
    map_intelligence_to_test_data(missing_indices, complete_matrix, student_intel)
    print('The accuracy of first model is %f' % sparse_matrix_evaluate(val_data, complete_matrix))
    # Here I should start


if __name__ == '__main__':
    enhancement_main()
