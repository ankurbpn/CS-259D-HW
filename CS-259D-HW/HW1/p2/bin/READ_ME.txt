We used a python script AnomalyDetect.py to implement the algorithm required for problem 2.
It contains 4 major functions.

Library Dependencies:
scipy
numpy
scikit-learn
pyplot
plyab
csv

get_pca_decompositions() finds pca decompositions for the entire data matrix read from the input files stored in the same folder ad outputs them to a file pca alongwith the features vectors for all the 50 users.
find_malicious_users() uses the pca decompositions and feature vectors to generate the layered networks and compare them over users to determine the similarity values and outputs them to a file results.csv.
find_best_threshold() reads the output of results.csv and works with several thresholds and outputs the result to threshold_finding.txt.
final_analysis() generates the vector for user21 and the individual false positive false negative values for all users for the chosen threshold value.

The code doesn't remove sparse commands so it needs a lot of memory (doesn't work on corn). We run our experiments on the madmax machine. 
The code has a lot of print statements so it would be advisable to pipe the output to a logfile.
