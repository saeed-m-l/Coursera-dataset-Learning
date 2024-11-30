from utils import *

import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, matrix):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_likelihood = 0.
    n_users = len(data['user_id'])
    n_questions = len(data['question_id'])
    # n_questions = len(data['question_id'])
    d_theta = np.zeros(n_users)
    users = data['user_id']
    questions = data['question_id']
    y = data['is_correct']
    d_beta = np.zeros(n_questions)
    for i in range(len(users)):
        tmp1 = users[i]
        tmp2 = questions[i]
        x = theta[tmp1] - beta[tmp2]
        p = 1 - sigmoid(-x)
        if p != 0 and p != 1:
            log_likelihood += y[i] * np.log(p) + (1 - y[i]) * np.log(1-p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta, matrix):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Here I assume data is matrix
    # n_users = len(data['user_id'])
    n_users = 542
    n_questions = 1774
    # n_questions = len(data['question_id'])
    d_theta = np.zeros(n_users)
    users = data['user_id']
    questions = data['question_id']
    y = data['is_correct']
    d_beta = np.zeros(n_questions)
    # for i in range(n_users):
    #     for j in range(n_questions):
    #         if matrix[i, j] == 0 or matrix[i, j] == 1:
    #             x = theta[i] - beta[j]
    #             p = sigmoid(-x)
    #             d_beta[j] += (matrix[i][j] - p)
    #             d_theta[i] += (p - matrix[i][j])
    # beta += lr * d_beta
    # theta += lr * d_theta
    for i in range(len(users)):
        tmp1 = users[i]
        tmp2 = questions[i]
        x = theta[tmp1] - beta[tmp2]
        p = 1 - sigmoid(-x)
        d_beta[tmp2] += (p - y[i])
        d_theta[tmp1] += (y[i] - p)
    beta += lr * d_beta
    theta += lr * d_theta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations, matrix):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    val_acc_lst = []
    neg_lld_lst = []
    theta = np.random.rand(542)
    beta = np.random.rand(1774)
    for i in range(iterations):
        (theta, beta) = update_theta_beta(theta=theta, beta=beta, data=data, lr=lr, matrix=matrix)
        val_acc_lst.append(evaluate(val_data, beta=beta, theta=theta))
        neg_lld_lst.append(neg_log_likelihood(data, theta=theta, beta=beta, matrix=matrix))
        lr *= 0.9
    # TODO: You may change the return values to achieve what you want.

    return theta, beta, val_acc_lst, neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def item_response_main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    print(sparse_matrix)
    val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")

    num_iterations = 300
    lr = 0.4
    learned_theta, learned_beta, val_acc_list, neg_lld_lst = irt(train_data, val_data, lr, num_iterations,
                                                                 sparse_matrix)
    x = np.array([i for i in range(1, 301)])
    plt.figure()
    plt.plot(x, val_acc_list)
    plt.xlabel('Number of iterations')
    plt.ylabel('Accuracy on validation set')
    plt.title('Accuracy in IRT')
    plt.savefig('../plots/IRT/irt.png')
    ind = np.argmax(np.array(val_acc_list))
    final_validation_acc = val_acc_list[ind]
    print('Best accuracy for itr = %d is %f' % (ind + 1, final_validation_acc))
    print('log likelihood is %f' % (neg_lld_lst[ind]))
    #####################################################################
    # Part B:                                                           #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    # -> Important Note: save plots instead of showing them!
    # I did that on upper part
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part C:                                                           #
    # Best Results                                                      #
    # -> Important Note: save plots instead of showing them!            #
    # I did that on upper part
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Plots                                                             #
    # -> Important Note: save plots instead of showing them!            #
    # based on explanation on report i will do this
    plt.figure()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for i in range(5):
        t = learned_theta[i]
        b = learned_beta[i]
        b = np.linspace(b - 2, b + 2, 100)
        x = np.exp(t - b) / (1 + np.exp(t - b))
        ax = axes[i]
        ax.plot(b, x)  # Replace x and y[i] with your data
        ax.set_title(f"{i + 1}")
        ax.set_xlabel('beta')
        ax.set_ylabel('probability')
    fig.delaxes(axes[5])
    fig.tight_layout()
    plt.savefig('../plots/IRT/partD.png')

    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
        'lr': lr,
        'num_iterations': num_iterations,
        'theta': learned_theta,
        'beta': learned_beta,
        'val_acc_list': val_acc_list,
        'neg_lld_lst': neg_lld_lst,
        'final_validation_acc': final_validation_acc,
    }
    return results


if __name__ == "__main__":
    item_response_main()
