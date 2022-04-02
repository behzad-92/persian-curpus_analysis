# -*- coding: utf-8 -*-
"""
@author: behzad
"""
from __future__ import unicode_literals
from string import punctuation
import numpy as np 
import pandas as pd
import os
import nltk
from hazm import *
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import svm

########### corpus_reader ##########################

def corpus_reader(the_path,):
    string_corpus = []
    the_path = the_path
    set_class = ''
    for filename in os.listdir(the_path):
        file_path = the_path+filename
        loaded_file = open(file_path,'r',encoding = "utf-8") 
        line_seperated_data = loaded_file.readlines() 
        set_class = str(filename[:2])
#         print(len(line_seperated_data))
        string_corpus.append([line_seperated_data[0],set_class])
    return string_corpus

############## input data ###############################
data= corpus_reader('/data/')
header= ['corpus','label']
Data = pd.DataFrame(data,columns=header)
Data.head()
pd.set_option('display.max_colwidth', -1)

###### pre process ######################################

punctuation = punctuation


def TextCleaner(Data, stopwordsList= ''):

    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    dataList = Data
    table = str.maketrans('', '', punctuation)
    CountVector = []
    for i in range(0, len(dataList)):
        vocabulary = []
        # vocabulary = Counter()
        for j in range(0, len(dataList[i][0])):

            dataList[i][0][j] = stemmer.stem(dataList[i][0][j])

            dataList[i][0][j] = lemmatizer.lemmatize(dataList[i][0][j])
        dataList[i][0] = [word for word in dataList[i][0] if word.isalpha()]
        dataList[i][0] = [w.translate(table) for w in dataList[i][0]]
        dataList[i][0] = [word for word in dataList[i][0] if len(word) > 3]

        vocabulary.append(dataList[i])


    return dataList

#### split data ################################
def String_Splitter(data):
    for i in range(len(data)):
         data[i][0]=data[i][0].split()
    return data

new_data = String_Splitter(data)

cleaned_data = TextCleaner(new_data)

########## convert split data to string ###############
def wordToString(wordList):
    stringList = []
    for i in range(0,len(wordList)):
        stringList.append(' '.join(word for word in wordList[i][0]))
    return stringList


last_string= wordToString(cleaned_data)

##################################
for i in range(len(data)):
    data[i][0]=last_string[i]
    

Final_Data = pd.DataFrame(data,columns=header)

le = preprocessing.LabelEncoder()
le.fit(Final_Data['label'])
le.classes_
Final_Data['labeled'] = le.transform(Final_Data['label'])

######## learning #######################

x_train, x_test, y_train, y_test = train_test_split(Final_Data.corpus, Final_Data.labeled, 
                                                    test_size=0.2, shuffle = True,  
                                                    random_state=0)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(Final_Data.corpus)
X_train_counts.shape


from scipy.sparse import csr_matrix
X_train_counts_Dense = X_train_counts.todense()

mean_list =  X_train_counts_Dense.mean(axis=0)
mean_list.argsort()[:10]

newDict = count_vect.vocabulary_

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

######## pipeline with NB Classifier  ################

from sklearn.pipeline import Pipeline

text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])

###################
text_clf.fit(x_train, y_train)    ### Training the model with a single command
predicted = text_clf.predict(x_test)
NB_accuracy= np.mean(predicted == y_test)  

##### pipeline with SGDClassifier ############
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                           max_iter=5, tol=None)),
 ])

text_clf.fit(x_train, y_train)  

###################
predicted = text_clf.predict(x_test)
SGD_accuracy= np.mean(predicted == y_test) 


from sklearn import metrics

print(metrics.classification_report(y_test, predicted))

############# svm ##########################

from sklearn.model_selection import cross_val_score
X = Final_Data.corpus
Y = Final_Data.labeled

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
X_train_counts.shape

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

clf = svm.SVC(kernel='linear', C=10)
scores = cross_val_score(clf, X_train_tf, Y, cv=5)

print(scores)
print(np.amax(scores))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


####### grid search ##############################

from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':range(1, 10)}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)

clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train_tf, Y)
sorted(clf.cv_results_.keys())

cvresults=clf.cv_results_

############## svc #########################
clf = svm.SVC(kernel='linear', C=10)
scores = cross_val_score(clf, X_train_tf, Y, cv=5)

print("SVC Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svc_scores=scores












