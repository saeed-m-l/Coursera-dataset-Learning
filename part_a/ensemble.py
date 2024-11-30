import copy

import numpy as np

#####################################################################
# TODO:                                                             #                                                          
# Import packages you need here                                     #
#####################################################################
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import *
import random


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  


#####################################################################
# Define and implement functions here                               #
#####################################################################
# This function do voting ( base on 3 base learner indices do voting e.g if we have two 1 and one zero result is 1
def create_voted_matrix(matrix1, matrix2, matrix3):
    # Perform voting to create a new matrix
    voted_matrix = np.zeros_like(matrix1)  # Initialize the voted matrix with zeros

    # Iterate over each cell in the matrices
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            # Count the number of 1s in the corresponding cells of the three matrices
            ones_count = matrix1[i, j] + matrix2[i, j] + matrix3[i, j]

            # Determine the value for the voted matrix based on the voting result
            voted_matrix[i, j] = 1 if ones_count >= 2 else 0

    return voted_matrix


#####################################################################
#                       END OF YOUR CODE                            #
##################################################################### 


def ensemble_main():
    #####################################################################
    # Compute the final validation and test accuracy                   #
    #####################################################################
    # TODO :
    test_acc_ensemble: float = None
    method1_output_matrix: np.array = None
    method2_output_matrix: np.array = None
    method3_output_matrix: np.array = None
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    train = load_train_csv("../data")
    train_n = len(train)
    # Here I have problem
    X1 = np.array(train['question_id'])
    X2 = np.array(train['user_id'])
    y = np.array(train['is_correct'])
    train_n = len(X1)
    x_train1 = []
    x_train2 = []
    x_train3 = []
    y_train1 = []
    y_train2 = []
    y_train3 = []
    for i in range(train_n):
        tmp1 = random.randrange(train_n)
        tmp2 = random.randrange(train_n)
        tmp3 = random.randrange(train_n)
        # print(tmp1, tmp2, tmp3)
        x_train1.append([X1[tmp1], X2[tmp1]])
        x_train2.append([X1[tmp2], X2[tmp2]])
        x_train3.append([X1[tmp3], X2[tmp3]])
        y_train1.append(y[tmp1])
        y_train2.append(y[tmp2])
        y_train3.append(y[tmp3])
    model1 = DecisionTreeClassifier(criterion='gini',
                                    max_depth=5,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
                                    max_features=None,
                                    random_state=42)
    model2 = DecisionTreeClassifier(criterion='gini',
                                    max_depth=5,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
                                    max_features=None,
                                    random_state=42)
    model3 = DecisionTreeClassifier(criterion='gini',
                                    max_depth=5,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
                                    max_features=None,
                                    random_state=42)
    model1.fit(x_train1, y_train1)
    model2.fit(x_train2, y_train2)
    model3.fit(x_train3, y_train3)
    # Fill the spare matrix
    spare = copy.deepcopy(sparse_matrix)
    method1_output_matrix = copy.deepcopy(sparse_matrix)
    method2_output_matrix = copy.deepcopy(sparse_matrix)
    method3_output_matrix = copy.deepcopy(sparse_matrix)
    missing_indices = np.where(np.isnan(spare))
    attributes = np.array(list(zip(missing_indices[0], missing_indices[1])))
    features = spare[:, [0, 1]]
    mask_matrix = np.zeros_like(features)
    mask_matrix[np.isnan(features)] = 1
    # Here i have problem
    predicted_values1 = model1.predict(attributes)
    predicted_values2 = model2.predict(attributes)
    predicted_values3 = model3.predict(attributes)
    #########
    per = model3.predict()
    method1_output_matrix[missing_indices] = predicted_values1
    method2_output_matrix[missing_indices] = predicted_values2
    method3_output_matrix[missing_indices] = predicted_values3
    ensemble_matrix = create_voted_matrix(method1_output_matrix, method2_output_matrix, method3_output_matrix)
    val_acc_ensemble = sparse_matrix_evaluate(val_data, ensemble_matrix)
    print('The accuracy of first model is %f' % sparse_matrix_evaluate(val_data, method1_output_matrix))
    print('The accuracy of second model is %f' % sparse_matrix_evaluate(val_data, method2_output_matrix))
    print('The accuracy of third model is %f' % sparse_matrix_evaluate(val_data, method3_output_matrix))
    print('The accuracy of voted is %f' % val_acc_ensemble)
    # Here I use validation set to get
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
        'val_acc_ensemble': val_acc_ensemble,
        'test_acc_ensemble': test_acc_ensemble,
        'method1_output_matrix': method1_output_matrix,
        'method2_output_matrix': method2_output_matrix,
        'method3_output_matrix': method3_output_matrix
    }

    return results


if __name__ == "__main__":
    ensemble_main()
