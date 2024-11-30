from .enhancement import enhancement_main



def student_information():
    #####################################################################
    # TODO:                                                             #
    # Please complete requested information                             #
    #####################################################################
    information = {
        'First_Name': 'Saeed',
        'Last_Name': 'Mansur lakuraj',
        'Student_ID': '99102304',
        'Submission_Date': '',  # In the Persian calendar [Khorshidi] format
    }
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return information


def test_student_code():
    results = {
        'Student_info': student_information(),
        'KNN': knn_main(),
        'item_response': item_response_main(),
        'matrix_factorization': matrix_factorization_main(),
        'ensemble': ensemble_main(),
    }

    return results
