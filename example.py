#!/bin/env python

import numpy as np
from scikits.learn.feature_extraction.text import WordNGramAnalyzer
from scikits.learn.linear_model.sparse import SGDClassifier
from scikits.learn import metrics
from scipy.sparse import coo_matrix, hstack, vstack
from vowpalwabbit import *
from pandas import *

def evaluate(clf,Xt,yt,Xv,yv,title):
  print title
  clf.fit(Xt,yt)
  pred = clf.predict(Xv)
  print metrics.classification_report(yv,pred)
 
if __name__ == "__main__":
  """Trivial example showing how to use the vowpal wabbit data munger:
  """
  vw = VowpalWabbit(analyzer=WordNGramAnalyzer(stop_words=set(), max_n=2, token_pattern='[^ ]+'), min_support=.001)

  #Load training data
  train_data,train_features_by_ns = vw.load_file("train.vw")
  vw.fit(train_features_by_ns)
  Xt_by_ns = vw.transform(train_features_by_ns)

  print "Tweets in training set containing apples"
  print train_data['meta'][['label','tag']][train_data['text']['apples'] > 0]

  print "Tweets in training set containing bananas"
  print train_data['meta'][['label','tag']][train_data['text']['bananas'] > 0]

 
  #Extract training labels 
  yt = np.asarray((train_data['meta']['label']),dtype=np.float32)

  #Extract train features for text, demographic and text+demographic
  Xt_t = Xt_by_ns['text']
  Xt_d = Xt_by_ns['demographic']
  Xt = hstack( [Xt_t,Xt_d] )

  #Load validation data
  val_data,test_features_by_ns = vw.load_file("val.vw")
  Xv_by_ns = vw.transform(test_features_by_ns)

  print "\nTweets in validation set containing apples"
  print val_data['meta'][['label','tag']][val_data['text']['apples'] > 0]

  print "Tweets in validation set containing bananas"
  print val_data['meta'][['label','tag']][val_data['text']['bananas'] > 0]
  
  #Extract validation labels 
  yv = np.asarray((val_data['meta']['label']),dtype=np.float32)

  #Extract val features for text, demographic and text+demographic
  Xv_t = Xv_by_ns['text']
  Xv_d = Xv_by_ns['demographic']
  Xv = hstack( [Xv_t,Xv_d] )

  #simple logistic classifier
  clf = SGDClassifier(alpha=.001, n_iter=20, penalty='elasticnet', rho='0.5')
  #clf = LogisticRegression(penalty='l2', eps=0.0001, C=1.0, fit_intercept=True)

  #evaluate on different feature groups
  print "\nModel Evaluation:\n"
  evaluate(clf,Xt_t,yt,Xv_t,yv,"Text features only")
  evaluate(clf,Xt_d,yt,Xv_d,yv,"Demographic features only")
  evaluate(clf,Xt,yt,Xv,yv,"All features")
