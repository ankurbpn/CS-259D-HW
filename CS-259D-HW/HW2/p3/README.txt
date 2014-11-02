Our analysis for this problem uses the script entropy_calculations.py .

The current setup will evaluate the sliding window entropies for all single columns and column pairs and save the output plots for each of them.
To evaluate the streaming entropies, uncomment the get_all_entropy_list() function, comment the windowed entropy function and change the filename in plot all entropies to what is given in the comment.

The get entropy functions read the log files to get the data (you'll need to add the log file to the folder) and store the entropy in a pickled object. The plot entropy function reads this output and saves *all* figures in the same folder.
