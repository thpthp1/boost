from numba.cuda.simulator import kernel
from svmutil import svm_problem, svm_parameter, svm_read_problem, svm_train, svm_predict
import numpy as np
import progressbar
from sklearn.ensemble import AdaBoostClassifier as ABClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def normalize(x): return (x - x.min()) / (np.ptp(x))

class AdaBoostClassifier:

    # Constructor for the AdaBoost Model taking in layering information, 
    # and defining two arrays with parameters / models
    def __init__(self, T):
        self.T = T
        self.clf = []
        self.alphas = []

    # Training function for the AdaBoost model.
    def train(self, X, y, tolerance=0.000001):

        Y = np.array(y)

        bar = progressbar.ProgressBar()
        weights = np.ones(len(X))/(len(X))

        # For each bar in a range of the iteration value T for the classifier
        for _ in bar(range(self.T)):
            # print(weights)
            # print(self.alphas)
            prob  = svm_problem(weights, y, X)
            param = svm_parameter('-t 0 -q')
            m_poly = svm_train(prob, param)
            self.clf.append(m_poly)
            p_label, p_acc, p_val = svm_predict(y, X, m_poly, '-q')
            labels = np.array(p_label).astype(np.float64)
            eps = weights[Y != labels].sum()
            #print(eps)
            if eps < tolerance:
                break
            alpha_t = 0.5 * np.log((1.0 - eps)/eps)
            self.alphas.append(alpha_t)
            weights = weights * np.exp(-Y * labels * alpha_t)
            weights /= np.sum(weights)

        # Convert alpha and clf info into np arrays.
        # self.alphas = np.asarray(self.alphas)
        # self.clf = np.asarray(self.clf)
    
    # Classify images with the adaboost classifier
    def classify(self, X):
        preds = np.zeros((1, len(X)))
    
        # The adaboost classification score is the sum of best weak classifier scores multiplied by the corresponding alpha
        for alpha, clf in zip(self.alphas, self.clf):
            weak_pred, _, _ = svm_predict([], X, clf, '-q')
            #print(weak_pred)
            preds += alpha * np.array(weak_pred)
        
        return np.sign(preds)

def load_data():
    y_test, x_test = svm_read_problem('./DogsVsCats/DogsVsCats.test')
    y_train, x_train = svm_read_problem('./DogsVsCats/DogsVsCats.train')
    return y_test, x_test, y_train, x_train

if __name__ == "__main__":
    y_test, x_test, y_train, x_train = load_data()
    y_sample, x_sample = svm_read_problem('./DogsVsCats/DogsVsCats.train', return_scipy=False)
    y_sample_test, x_sample_test = svm_read_problem('./DogsVsCats/DogsVsCats.test', return_scipy=False)
    # print(len(y_train))
    # print(len(x_train))
    model = AdaBoostClassifier(20)
    model.train(x_sample, y_sample)
    preds = model.classify(x_test)
    print(np.sum(preds == np.array(y_test))/len(y_test))
    # vals = []
    # for i in range(1, 11):
    #     model = AdaBoostClassifier(i)
    #     model.train(x_sample, y_sample)
    #     preds = model.classify(x_test)
    #     vals.append(np.sum(preds == np.array(y_test))/len(y_test))
    # plt.plot(np.arange(1, 11), vals)
    # plt.show()
    # y_sample, x_sample = svm_read_problem('./DogsVsCats/DogsVsCats.train', return_scipy=True)
    # y_sample_test, x_sample_test = svm_read_problem('./DogsVsCats/DogsVsCats.test', return_scipy=True)
    # model = ABClassifier(base_estimator=SVC(kernel='poly', degree=5), algorithm='SAMME', n_estimators=10)
    # model.fit(x_sample.todense(), y_sample)
    # print(model.score(x_sample_test.todense(), y_sample_test))

    