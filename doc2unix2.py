#!/usr/bin/env python3
"""
convert dos linefeeds (crlf) to unix (lf)
usage: python dos2unix.py
"""

# Sample usage of try catch in code 

# import sys
# import pickle
# sys.path.append('../tools/')
# # sys.path.append('../tools/')
 
# from doc2unix2 import doc2unix2

# file_path = '../17-final-project/final_project_dataset.pkl'

# try:
#     with open(file_path, 'rb') as f:
#         enron_data = pickle.load(f)
        
# except pickle.UnpicklingError:
#     file_path = doc2unix2(file_path)
#     with open(file_path, 'rb') as f:
#         enron_data = pickle.load(f)



import sys

def doc2unix2(file_name):
    # original = '../tools/word_data.pkl'
    # destination = "../tools/word_data_unix.pkl"

    content = ''
    outsize = 0
    with open(file_name, 'rb') as infile:
        content = infile.read()
    with open(file_name, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    print("Done. Saved %s bytes." % (len(content)-outsize))
    return file_name