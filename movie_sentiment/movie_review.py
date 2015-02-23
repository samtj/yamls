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
import numpy as np
import re

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
    test_set = []

    ##############################TRAIN DATA#####################
    with open(data_path + 'train.tsv', 'r') as f:
        sentimentreader = csv.reader(f, delimiter='\t')
        header = sentimentreader.next()
        cnt = 0
        for row in sentimentreader:
            sentence = row[2]
            sentiment = row[3]
            if (cnt < 30):
                test_set.append((sentence, sentiment))
            elif (cnt > 30):
                #cl.train(sentence, sentiment)
                train_set.append((sentence, sentiment))
            cnt += 1
    
    sentence_set = [sentence for sentence, label in train_set]
    label_set = [int(label) for sentence, label in train_set]

    test_sentence = [sentence for sentence, label in test_set]
    label_test = [int(label) for sentence, label in test_set]

    text_clf = scikit_learn(sentence_set, label_set)
    
    #Predict the test data
    #predicted = text_clf.predict(test_sentence)
    #print np.mean(predicted == np.asarray(label_test))
    #for doc, category in zip(test_sentence, predicted):
    #    print('%r => %s' % (doc, category))
    #cl = NaiveBayesClassifier.train(train_set)

    #############################TEST DATA##################
    
    # Read test data and predict phrase based on train set
    with open(data_path + 'test.tsv', 'r') as f:
        testreader = csv.reader(f, delimiter='\t')
        submission = open('scikit_submission.csv', 'w')
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
    return

def scikit_learn(train_set, train_labels):
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsOneClassifier(LinearSVC())),
                     ])
    X_train = np.asarray(train_set)
    text_clf = text_clf.fit(X_train, np.asarray(train_labels))
    return text_clf

if __name__ == '__main__':
    process_csv()
    #scikit_learn(train_set)