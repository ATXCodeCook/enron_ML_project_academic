# enron_ML_project_academic
Machine learning in Python with SciKit Learn for academic project

Original project as been updated from Python 2 to Python 3

Converted some files to Python 3.8 syntax (listed below):

tester.py:
Update to Python 3.8:
1. Changed print '...'  to print('...')
2. Updated path to StratifiedShuffleSplit to sklearn.selection
3. Changed StratifiedShuffleSplit call to updated version (drop label parameter)
4. Changed pickle.dump from 'w' to 'wb'
5. Changed pickle.load from 'r' to 'rb'
6. Added dictionary to return scores for use in df
