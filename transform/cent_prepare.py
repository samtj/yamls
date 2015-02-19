__author__ = 'samueltjokrosoesilo'

from math import *

import numpy as np
import scipy.linalg as linalg
import arff
import imp

csvhelper = imp.load_source('csv', 'helper/csv.py')


# helper to prepare data for supervised learning
# generates both csv and arff files

# csv can be consumed using scikit-learn Python
# arff can be imported into Weka for analysis

def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def filterbyvalue(seq, x, y):
    for el in seq:
        if (el[0] != x and el[1] != y): yield el
        
        
folder = '../data/centroid/'
title = 'centroid'
type = 'xform'

# 25828 rows
# train 0.7
# validation 0.3
# test 1000
split = 0.7
test = 1000

data = np.loadtxt((folder + title + '.csv'),delimiter=',')
target = data[:, -1]
flat_data = data[:, :-1]

# 854, 480

# clean up -1,-1 noise
clean = [elem for elem in data if elem[0] != -1. and elem[1] != -1.]

print 'all:', len(clean)
contentX = []
contentY = []
# previous 2 headings, current position
for i in range(len(clean)-3):
    heading1 = get_heading(clean[i], clean[i+1])
    heading2 = get_heading(clean[i+1], clean[i+2])
    #position = int((clean[i+2][1]-1) * 480 + clean[i+2][0])
    #nextPosition = int((clean[i+3][1]-1) * 480 + clean[i+3][0])
    
    # 2 points to give velocity per frame
    pX1 = int(clean[i+1][0])
    pX2 = int(clean[i+2][0])
    pY1 = int(clean[i+1][1])
    pY2 = int(clean[i+2][1])

    # target point
    nX = int(clean[i+3][0])
    nY = int(clean[i+3][1])

    contentX.append([heading1, heading2, pX1, pX2, nX])
    contentY.append([heading1, heading2, pY1, pY2, nY])
    
    #content.append([position,heading1,heading2,nextPosition])
    #print position, heading1, heading2, nextPosition

testX = contentX[-test:]
testY = contentY[-test:]
trainX = contentX[:]
trainY = contentY[:]

# write to arff files
# this is for Weka learning
cNamesX = ['heading1', 'heading2', 'px1', 'px2']
cNamesX.append('@@class')

cNamesY = ['heading1', 'heading2', 'py1', 'py2']
cNamesY.append('@@class')

# 24348
# train/validation 23348
# test 1000

print 'train/valid:', len(contentX)


# CSV
csvhelper.writetocsv(folder, title + '_' + type + '_x.csv', contentX)
csvhelper.writetocsv(folder, title + '_' + type + '_y.csv', contentY)

# train/valid
csvhelper.writetocsv('', folder + title + '_' + type + '_train_x.csv', trainX)
csvhelper.writetocsv('', folder + title + '_' + type + '_train_y.csv', trainY)

print 'test:', len(testX)

# test
csvhelper.writetocsv('', folder + title + '_' + type + '_test_x.csv', testX)
csvhelper.writetocsv('', folder + title + '_' + type + '_test_y.csv', testY)


# x,y
train = clean[:-(test)]
test = clean[-(test):]

# all
csvhelper.writetocsv('', folder + title + '_clean.csv', clean)

# train, test
csvhelper.writetocsv('', folder + title + '_clean_train.csv', train)
csvhelper.writetocsv('', folder + title + '_clean_test.csv', test)


# write to arff files
cNamesX = ['heading1', 'heading2', 'px1', 'px2']
cNamesX.append('@@class')

cNamesY = ['heading1', 'heading2', 'py1', 'py2']
cNamesY.append('@@class')

# for the final submission, we don't need to generate arff files for Weka
# ARFF for X
#output = arff.Writer(title + '_x.arff', relation=title, names=cNamesX)
#for row in contentX:
#    output.write(row)
#output.close()    

# ARFF for Y
#output = arff.Writer(title + '_y.arff', relation=title, names=cNamesY)
#for row in contentY:
#    output.write(row)
#output.close()   