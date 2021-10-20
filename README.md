# enron_ML_project_academic
Machine learning in Python with SciKit Learn for academic project

Note: Original project has been updated from Python 2 to Python 3

This objective of this project is to create a machine learning algorithm
that will identify employees that were involved in the Enron fraud (prosecuted)
based on financial, social and other company data.

Updates:

07/14/2021
Converted some files to Python 3.8 syntax (listed below):

tester.py:
Update to Python 3.8:
1. Changed print '...'  to print('...')
2. Updated path to StratifiedShuffleSplit to sklearn.selection
3. Changed StratifiedShuffleSplit call to updated version (drop label parameter)
4. Changed pickle.dump from 'w' to 'wb'
5. Changed pickle.load from 'r' to 'rb'
6. Added dictionary to return scores for use in df
