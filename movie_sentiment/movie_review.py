# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

import csv
#from textblob.classifiers import NaiveBayesClassifier
#from nltk import NaiveBayesClassifier, classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

import numpy as np
import re
#import helper.csvhelper
#import arff
from sklearn import cross_validation

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

def scikit_test(text_clf):
    with open(data_path + 'test.tsv', 'r') as f:
        testreader = csv.reader(f, delimiter='\t')
        submission = open(data_path + 'sam_svc_submission.csv', 'wb')
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
        
        predicted = text_clf.predict(np.asarray(phrase_list))
        for i in range(len(predicted)):
            sentiment_label = predicted[i]
            phraseid = phraseid_list[i]
            csvwriter.writerow([str(phraseid), str(sentiment_label)])
        print 'done'
    return

def scikit_learn(train_set, train_labels, CV):
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC

    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
#                     ('clf', RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)),
                     ])

#                     ('clf', OneVsOneClassifier(RandomForestClassifier())),
#                     ('clf', RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=1, random_state=0)),
#                     ('clf', svm.SVC(kernel='linear')),
#                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5)),
#                     ('clf', MultinomialNB()),
#                     ('clf', OneVsOneClassifier(LinearSVC())),
    X = np.asarray(train_set)
    y = np.asarray(train_labels)
    
    if CV:
        kf = cross_validation.KFold(len(X), n_folds=5)
        
        scores = []
        for k, (train, test) in enumerate(kf):
            text_clf.fit(X[train], y[train])
            score = text_clf.score(X[test], y[test])
            print("[fold {0}], score: {1:.5f}".format(k, score ))
            scores.append(score)
    #        scores.append()
        print np.mean(scores)
        text_clf = text_clf.fit(X, np.asarray(y))
    else:
        text_clf = text_clf.fit(X, np.asarray(y))
    return text_clf

if __name__ == '__main__':
    train_set, train_label = process_csv()
    text_clf = scikit_learn(train_set, train_label, CV=False)
    scikit_test(text_clf)
    