from svmutil import svm_train, svm_problem, svm_parameter, svm_predict, svm_read_problem
import os
import numpy as np

def load_data():
    y_test, x_test = svm_read_problem('./DogsVsCats/DogsVsCats.test')
    y_train, x_train = svm_read_problem('./DogsVsCats/DogsVsCats.train')
    return y_test, x_test, y_train, x_train

if __name__ == '__main__':
    y_test, x_test, y_train, x_train = load_data()

    # Training
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-t 0 -v 10')
    m_lin = svm_train(prob, param)

    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-t 1 -d 5 -v 10')
    m_poly = svm_train(prob, param)