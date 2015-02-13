__author__ = 'samueltjokrosoesilo'

import csv
from os.path import join

# helper to write to csv file
def writetocsv(path, fileName, dataset):
    with open(join(path, fileName), 'wb') as fp:
        oWriter = csv.writer(fp, delimiter=',')
        oWriter.writerows(dataset)