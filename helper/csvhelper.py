import csv
from os.path import join


def writetocsv(path, fileName, dataset):
    with open(join(path, fileName), 'wb') as fp:
        oWriter = csv.writer(fp, delimiter=',')
        oWriter.writerows(dataset)