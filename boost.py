from svmutil import *
import numpy as np
import progressbar


class AdaBoostClassifier:

    # Constructor for the AdaBoost Model taking in layering information, 
    # and defining two arrays with parameters / models
    def __init__(self, T):
        self.T = T
        self.clf = []
        self.alphas = []

    # Training function for the AdaBoost model.
    def train(self, X, y):

        bar = progressbar.ProgressBar()

        # For each bar in a range of the iteration value T for the classifier
        for _ in bar(range(self.T)):
            # Reduce the weights by their norms
            weights /= np.linalg.norm(weights)

            # Create weak classifiers from the current images / features
            

            # Find the best classifier
            results = np.array(results)

            # If we converge
            if min_error == 0 or min_error > 0.5:
                bar.finish()
                break

            # Calculate our modification factor beta
            beta = min_error/(1-min_error)
            beta_pow = np.power(beta, 1 - results)

            # Modify the weights for next iteration
            weights = weights * beta_pow

            # Create our alpha
            alpha = np.log(1.0/beta)

            # Remember the alpha in the classifier
            self.alphas.append(alpha)
            # Remember the best weak classifier in this itteration
            self.clf.append(best_clf)

        # Convert alpha and clf info into np arrays.
        self.alphas = np.asarray(self.alphas)
        self.clf = np.asarray(self.clf)
    
    # Classify images with the adaboost classifier
    def classify(self, X):
        
        # The adaboost classification score is the sum of best weak classifier scores multiplied by the corresponding alpha
        classify_score = np.sum([alpha * clf.classify(integral_image) for alpha, clf in zip(self.alphas, self.clf)])
        
        # Define a random threshold for classification based on the sum of alphas
        random_thresh = 0.5 * np.sum(self.alphas)

        return 1 if classify_score >= random_thresh else 0