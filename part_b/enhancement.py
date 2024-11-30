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
from utils import *


def create_voted_vec(vec1, vec2, vec3):
    # Perform voting to create a new matrix
    voted_matrix = np.zeros_like(vec1)  # Initialize the voted matrix with zeros

    # Iterate over each cell in the matrices
    for i in range(len(vec1)):
        # Count the number of 1s in the corresponding cells of the three matrices
        ones_count = (vec1[i] + vec2[i] + vec3[i])/2.5

        # Determine the value for the voted matrix based on the voting result
        voted_matrix[i] = 1 if ones_count >= 2 else 0

    return voted_matrix


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


def map_intelligence_to_test_data(test_student_ids, train_intelligence, max=1.5, min=0.75):
    test_intelligence = np.zeros(len(test_student_ids))

    for i, student_id in enumerate(test_student_ids):
        train_indices = np.where(train_intelligence[:, 0] == student_id)[0]

        if len(train_indices) > 0:
            train_int = train_intelligence[train_indices[0], 1]

            if train_int >= 0.5:
                test_intelligence[i] = max
            elif train_int < 0.5:
                test_intelligence[i] = min
            else:
                test_intelligence[i] = 1
        else:
            test_intelligence[i] = 1

    return test_intelligence


def preprocess_data(data):
    question_id = data[:, 0]
    user_id = data[:, 1]
    subject_id = data[:, 2]

    # Perform one-hot encoding on subject_id
    mlb = MultiLabelBinarizer()
    subject_id_encoded = mlb.fit_transform(subject_id)

    # Combine the encoded subject_id with user_id and question_id
    processed_data = np.column_stack((user_id, question_id, subject_id_encoded))
    return processed_data, mlb


def add_subject(data, question, subject):
    number = len(data)
    lis_subject = []
    for i in range(number):
        index = question.index(data[i])
        lis_subject.append(subject[index])
    return lis_subject


def preprocess_test_data(test_data, mlb):
    question_id = test_data[:, 0]
    user_id = test_data[:, 1]
    subject_id = test_data[:, 2]

    subject_id_encoded = mlb.transform(subject_id)

    processed_test_data = np.column_stack((user_id, question_id, subject_id_encoded))

    return processed_test_data


def enhancement_main():
    # I will add gender to it
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    subject = load_subject_csv("../data")
    train = load_train_csv("../data")
    # Here at first I should add data
    # then i should use PCA
    # I should find the best value for K
    # TODO one hot
    # TODO PCA
    X1 = np.array(train['question_id'])
    X2 = np.array(train['user_id'])
    X1_val = np.array(val_data['question_id'])
    X2_val = np.array(val_data['user_id'])
    lis_topic = add_subject(X1, subject['question_id'], subject['subject_id'])
    lis_topic_val = add_subject(X1_val, subject['question_id'], subject['subject_id'])
    y = np.array(train['is_correct'])
    y_val = np.array(val_data['is_correct'])
    train_n = len(X1)
    x_train1 = []
    x_val = []
    for i in range(train_n):
        # x_train1.append([X1[i], X2[i]])
        x_train1.append([X1[i], X2[i], lis_topic[i]])
    for i in range(len(X2_val)):
        # x_val.append([X1_val[i], X2_val[i]])
        x_val.append([X1_val[i], X2_val[i], lis_topic_val[i]])
    # Here add One Hot encoder
    student_intel = get_intelligence(X2, y)
    question_intel = get_intelligence(X1, y)
    val_student_intel = map_intelligence_to_test_data(X2_val, student_intel)
    val_question_intel = map_intelligence_to_test_data(X1_val, question_intel)
    x_val = np.array(x_val)
    x_train1 = np.array(x_train1)
    # Here I should start
    processed_data, mlb = preprocess_data(x_train1)
    processed_val_data = preprocess_test_data(x_val, mlb)
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(processed_data)
    x_val_pca = pca.transform(processed_val_data)
    # Rejected Algorithm
    # dt = DecisionTreeClassifier(max_depth=1)
    # ab = AdaBoostClassifier(base_estimator=dt, n_estimators=1000)
    # # bg = BaggingClassifier(base_estimator=ab, n_estimators=10, random_state=42)
    # ab.fit(X_train_pca, y)
    #
    # y_pred = ab.predict(X_train_pca)
    # print(accuracy_score(y, y_pred))
    knn = KNeighborsRegressor(n_neighbors=5)
    tree1 = DecisionTreeRegressor(max_depth=5,
                                  min_samples_split=10,
                                  min_samples_leaf=5,
                                  max_features=None,
                                  random_state=42)
    tree2 = DecisionTreeRegressor(max_depth=10,
                                  min_samples_split=8,
                                  min_samples_leaf=4,
                                  max_features=None,
                                  random_state=42)
    # Define the bagging classifier with KNN as the base estimator
    # bagging = BaggingClassifier(base_estimator=knn, n_estimators=10, random_state=42)

    # Fit the ensemble model on the training data
    tree1.fit(X_train_pca, y)
    tree2.fit(X_train_pca, y)
    knn.fit(X_train_pca, y)
    # Make predictions on the test data
    val_student_intel = map_intelligence_to_test_data(X2_val, student_intel, 2, 0.6)
    y_pred_tree1 = tree1.predict(x_val_pca)
    #y_pred_tree1 = multiply_and_threshold(y_pred_tree1, val_student_intel)
    # y_pred = threshold_processing(y_pred, val_student_intel, val_question_intel)
    # Calculate accuracy
    accuracy = accuracy_score(multiply_and_threshold(y_pred_tree1, val_student_intel), y_val)
    #print("Accuracy of tree 1 is:", accuracy)
    y_pred_knn = knn.predict(x_val_pca)
    #accuracy = accuracy_score(multiply_and_threshold(y_pred_knn, val_student_intel), y_val)
    #print("Accuracy of knn is :", accuracy)
    y_pred_tree2 = tree2.predict(x_val_pca)
    #accuracy = accuracy_score(multiply_and_threshold(y_pred_tree2, val_student_intel), y_val)
    #print("Accuracy of tree 2 is:", accuracy)
    #voted = create_voted_vec(y_pred_tree2, y_pred_tree1, y_pred_knn)
    voted = (y_pred_tree2+y_pred_knn+y_pred_tree1)/3
    acc_voted = accuracy_score(y_val, multiply_and_threshold(voted, val_student_intel))
    print("Accuracy of voted is :", acc_voted)
    # This part is for test ( load test then you can run it)
    '''
    # Here load dict test ( like validation)
    test = ...
    X1_test = np.array(test_data['question_id'])
    X2_test = np.array(test_data['user_id'])
    for i in range(len(X2_test)):
        x_test.append([X1_test[i], X2_test[i], lis_topic_test[i]])
    x_test = np.array(x_test)
    processed_test_data = preprocess_test_data(x_test, mlb)
    x_test_pca = pca.transform(processed_test_data)
    y_pred_knn = knn.predict(x_test_pca)
    y_pred_tree2 = tree2.predict(x_test_pca)
    y_pred_tree1 = tree1.predict(x_test_pca)
    voted = (y_pred_tree2+y_pred_knn+y_pred_tree1)/3
    acc_voted = accuracy_score(y_test, multiply_and_threshold(voted, test_student_intel))
    print("Accuracy of voted test is is :", acc_voted)
    '''

if __name__ == '__main__':
    enhancement_main()
