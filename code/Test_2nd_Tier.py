execfile("Utils.py")
import Utils
import cPickle as pickle
import os
import sys
import numpy as np
import random
from sklearn import svm, metrics
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Model_2nd_Tier
  model_2nd_tier = pickle.load(open(os.path.join(path_pickles, 'Model_2nd_Tier.pickle'), 'r'))
  # Load X
  X = pickle.load(open(os.path.join(path_pickles, 'X_2nd_Tier.pickle'), 'r'))
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_2nd_Tier.pickle'), 'r'))

  # Prediction
  X_n = []
  y_n = []
  X_y = []
  y_y = []
  for i in range(y.shape[0]):
    temp_x = list(X[i])
    temp_y = int(y[i])
    if temp_y == 1:
      X_y.append(temp_x)
      y_y.append(temp_y)
    else:
      X_n.append(temp_x)
      y_n.append(temp_y)
  X_train = X_y
  y_train = y_y
  
  # Put X_n and y_n together
  Xy_n = []
  for i in range(len(X_n)):
    Xy_n.append((X_n[i], y_n[i]))

  # Pick randomly from Xy_n
  times = 1
  sample = random.sample(Xy_n, int(len(y_train)*times))
  for (d, a) in sample:
    X_train.append(d)
    y_train.append(a)
  
  # Perform leave-one-out evaluation
  total = len(X_train)
  correct = 0
  svm_predicted = []
  svm_expected = []
  naivebayes_predicted = []
  naivebayes_expected = []
  decisiontree_predicted = []
  decisiontree_expected = []
  for i in range(len(X_train)):
    front_X = X_train[:i]
    test_X = X_train[i]
    back_X = X_train[i+1:]
    
    front_y = y_train[:i]
    test_y = y_train[i]
    back_y = y_train[i+1:]

    training_X = front_X + back_X
    training_y = front_y + back_y
    training_X = np.asarray(training_X)
    training_y = np.asarray(training_y)

    clf = svm.SVC()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    svm_predicted.append(int(predicted[0]))
    svm_expected.append(int(expected))
    
    clf = GaussianNB()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    naivebayes_predicted.append(int(predicted[0]))
    naivebayes_expected.append(int(expected))
    
    clf = DecisionTreeClassifier()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    decisiontree_predicted.append(int(predicted[0]))
    decisiontree_expected.append(int(expected))
  svm_predicted = np.asarray(svm_predicted)
  svm_expected = np.asarray(svm_expected)
  print metrics.classification_report(svm_expected, svm_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(svm_expected, svm_predicted)
  naivebayes_predicted = np.asarray(naivebayes_predicted)
  naivebayes_expected = np.asarray(naivebayes_expected)
  print metrics.classification_report(naivebayes_expected, naivebayes_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(naivebayes_expected, naivebayes_predicted)
  decisiontree_predicted = np.asarray(decisiontree_predicted)
  decisiontree_expected = np.asarray(decisiontree_expected)
  print metrics.classification_report(decisiontree_expected, decisiontree_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(decisiontree_expected, decisiontree_predicted)
  
  print
  print


  # Perform n-fold cross validation
  # Total number = 56
  # We do 14-fold, each fold has 52 train, 4 test
  total = len(X_train)
  svm_predicted = []
  svm_expected = []
  naivebayes_predicted = []
  naivebayes_expected = []
  decisiontree_predicted = []
  decisiontree_expected = []
  for i in range(total/4):
    front_X = X_train[:i*4]
    test_X = X_train[i*4:i*4+4]
    back_X = X_train[i*4+4:]
    
    front_y = y_train[:i*4]
    test_y = y_train[i*4:i*4+4]
    back_y = y_train[i*4+4:]
    
    training_X = front_X + back_X
    training_y = front_y + back_y
    training_X = np.asarray(training_X)
    training_y = np.asarray(training_y)

    clf = svm.SVC()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    svm_predicted.extend(predicted)
    svm_expected.extend(expected)
    
    clf = GaussianNB()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    naivebayes_predicted.extend(predicted)
    naivebayes_expected.extend(expected)
    
    clf = DecisionTreeClassifier()
    clf.fit(training_X, training_y)
    predicted = clf.predict(test_X)
    expected = test_y
    decisiontree_predicted.extend(predicted)
    decisiontree_expected.extend(expected)
  svm_predicted = np.asarray(svm_predicted)
  svm_expected = np.asarray(svm_expected)
  print metrics.classification_report(svm_expected, svm_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(svm_expected, svm_predicted)
  naivebayes_predicted = np.asarray(naivebayes_predicted)
  naivebayes_expected = np.asarray(naivebayes_expected)
  print metrics.classification_report(naivebayes_expected, naivebayes_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(naivebayes_expected, naivebayes_predicted)
  decisiontree_predicted = np.asarray(decisiontree_predicted)
  decisiontree_expected = np.asarray(decisiontree_expected)
  print metrics.classification_report(decisiontree_expected, decisiontree_predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(decisiontree_expected, decisiontree_predicted)
