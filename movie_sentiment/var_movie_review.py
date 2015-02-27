# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

import csv
#from textblob.classifiers import NaiveBayesClassifier
#from nltk import NaiveBayesClassifier, classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

import numpy as np
import re
import conv.convnet as cn
from sklearn import cross_validation
from sklearn_theano.feature_extraction import OverfeatClassifier
import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from helper.sklearn_api import MaxoutClassifier
from helper.sklearn_api import Classifier
from scipy import sparse
import cPickle as pickle
import os.path

data_path = '../data/movie_sentiment/'

def getwords(doc):
    """docstring for getwords"""
    splitter=re.compile('\\W*')
    words=[s.lower() for s in splitter.split(doc)
            if len(s) > 2 and len(s) < 20]

    return dict([(w, 1) for w in words])

def get_features(item):
    words = getwords(item)
    return dict((word, 'true') for word in words)

def get_accuracy(cl, test_set):
    success = 0
    for feature, label in test_set:
        guess = cl.classify(feature)
        if guess == label:
            success+=1
    return float(success/len(test_set))

def process_csv():
    # Train the classifier
    train_set = []

    ##############################TRAIN DATA#####################
    with open(data_path + 'train.tsv', 'r') as f:
        sentimentreader = csv.reader(f, delimiter='\t')
        header = sentimentreader.next()
        cnt = 0
        for row in sentimentreader:
            sentence = row[2]
            sentiment = row[3]
            train_set.append((sentence, sentiment))
            cnt += 1
    
    sentence_set = [sentence for sentence, label in train_set]
    label_set = [int(label) for sentence, label in train_set]
    """

    cNamesTr = ['sentence']
    cNamesTr.append('@@class')
    
    # ARFF for Train
    output = arff.Writer(data_path + 'train.arff', names=cNamesTr)
    for row in train_set:
        output.write(row)
    output.close()    

    output = arff.Writer(data_path + 'test.arff', names=cNamesTr)
    for row in test_set:
        output.write(row)
    output.close()    

    #csvhelper.writetocsv('', title + '_' + type + '_x.csv', contentX)
    """ 
    return sentence_set, label_set


    text_clf = scikit_learn(sentence_set, label_set)
    #Predict the test data
    #predicted = text_clf.predict(test_sentence)
    #print np.mean(predicted == np.asarray(label_test))
    #for doc, category in zip(test_sentence, predicted):
    #    print('%r => %s' % (doc, category))
    #cl = NaiveBayesClassifier.train(train_set)

    #############################TEST DATA##################
    
    # Read test data and predict phrase based on train set

def scikit_test(text_clf):    
    with open(data_path + 'test.tsv', 'r') as f:
        testreader = csv.reader(f, delimiter='\t')
        submission = open('sam_pylearn_submission.csv', 'wb')
        csvwriter = csv.writer(submission, delimiter=',')
        csvwriter.writerow(['PhraseId', 'Sentiment'])
        header = testreader.next()
        phraseid_list = []
        phrase_list = []
        for row in testreader:
            phraseid = row[0]
            phrase = row[2]
            phraseid_list.append(phraseid)
            phrase_list.append(phrase)
            #rating = cl.classify(phrase, default='0')
            #write_row = [str(phraseid), str(rating)]
            #csvwriter.writerow(write_row)
        
        #X_train_counts = cv.fit_transform(np.asarray(phrase_list))
        #X_train_tf = tf.transform(X_train_counts)
        
        test_list = np.asarray(phrase_list)
        print test_list.shape
        print test_list[0].shape
        predicted = text_clf.predict(test_list)
        for i in range(len(predicted)):
            sentiment_label = predicted[i]
            phraseid = phraseid_list[i]
            csvwriter.writerow([str(phraseid), str(sentiment_label)])
        print 'done'
    return

def scikit_learn(train_set, train_labels, CV):
#    count_vect = CountVectorizer()
#    X_train_counts = count_vect.fit_transform(np.asarray(train_set))
#    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#    X_train_tf = tf_transformer.transform(X_train_counts)
                                          
#    clsfr = cn.get_conv(X_train_tf.shape[1], 5)
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MaxoutClassifier()),
#                     ('clf', OneVsOneClassifier(LinearSVC())),
                     ])

#                     ('clf', svm.SVC()),
#                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5)),
#                     ('clf', MultinomialNB()),
#                     ('clf', OneVsOneClassifier(LinearSVC())),
    
    X = np.asarray(train_set)
    y = np.asarray(train_labels)
#    X = X_train_tf
#    y = np.asarray(train_labels)
    
#    y = np.zeros((labels.shape[0],5), dtype=np.int)
    
#    for i in range(labels.shape[0]):
#        y[i][labels[i]] = 1
    
#    print y.shape
#    text_clf = OverfeatClassifier()
    
    if CV:
        kf = cross_validation.KFold(X.shape[0], n_folds=5)
        
        scores = []
        for k, (train, test) in enumerate(kf):
            text_clf.fit(X[train], y[train])
#            score = text_clf.predict(X[test])
            score = text_clf.score(X[test], y[test])
            print score
            #print("[fold {0}], score: {1:.5f}".format(k, score ))
            scores.append(score)
    #        scores.append()
        print np.mean(scores)
        text_clf = text_clf.fit(X, y)
    else:
        print X.shape
        print X[0].shape
        print y.shape
        print y[0].shape

        # if train's already done
        if os.path.exists('pylearn2.pickle'):
            with open(r"pylearn2.pickle", "rb") as input_file:
                net2 = pickle.load(input_file)
                text_clf = net2
        else:
            text_clf.fit(X, y)
            with open('pylearn2.pickle', 'wb') as f:
                pickle.dump(text_clf, f, -1)
    return text_clf

if __name__ == '__main__':
    train_set, train_label = process_csv()
    text_clf = scikit_learn(train_set, train_label, CV=False)
    scikit_test(text_clf)