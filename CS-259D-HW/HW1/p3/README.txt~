Problem 3 discussion
--------------------

Our initial approach was to take a relatively simple detector and train it with all the feature vectors to see how it would perform. For this we chose Manhattan distance with the mean and variance of the 1-norm of the training data used to determine the threshold value for each user. We noticed that we could reduce the size of our feature vectors by removing one of three features for each key (either Hold, Down-down, or Up-down), since any third feature can be determined by the other two. However, we chose not to remove redundant features as they would not actually affect the outcome of the classifier, since it would scale every vector and the threshold values by a constant factor.

We see that our classifier has a consistently high true positive rate, but a moderately high false positive rate as well (where we define a true positive to be a user labeled as authentic when the user is indeed authentic, and a false positive to be a masquerader labeled as authentic). For example, the classifier would achieve a 88% true positive at roughly a 50% false positive rate.

---
TODO: PCA analysis discussion
---
