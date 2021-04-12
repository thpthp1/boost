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
    W = W = [1] * len(y_train)
    prob  = svm_problem(W, y_train, x_train)
    param = svm_parameter('-t 0')
    m_lin = svm_train(prob, param)

    prob  = svm_problem(W, y_train, x_train)
    param = svm_parameter('-t 1 -d 5')
    m_poly = svm_train(prob, param)

    p_label, p_acc, p_val = svm_predict(y_test, x_test, m_poly)
    print(p_label)