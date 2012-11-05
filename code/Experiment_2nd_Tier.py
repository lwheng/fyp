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

  # Load X
  X = pickle.load(open(os.path.join(path_pickles, 'X_2nd_Tier_Best.pickle'), 'r'))
  X = np.asarray(X)
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_2nd_Tier_Best.pickle'), 'r'))
  y = np.asarray(y)

  # Feature Ablation
  correct = 0
  total = y.shape[0]
  expected = y
  # Full
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X, y)
  predicted = clf.predict(X)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "Full : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Remove first feature
  X_train = []
  for x in X:
    X_train.append(x[2:])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "Full - 1st : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Remove second feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[0:2] + x[3:])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "Full - 2nd : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Remove third feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[0:3] + x[4:])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "Full - 3rd : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Remove fourth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[0:4])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "Full - 4th : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Only first feature
  X_train = []
  for x in X:
    X_train.append(x[0:2])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "1st : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Only second feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append([x[2]])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "2nd : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Only third feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append([x[3]])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "3rd : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Only fourth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append([x[4]])
  X_train = np.asarray(X_train)
  correct = 0
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y)
  predicted = clf.predict(X_train)
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  accuracy = float(correct) / float(total)
  print "4th : " + str(correct) + " / " + str(total) + " = " + str(accuracy)

  print
  # Leave-One-Out
  X = list(X)
  y = list(y)
  total = len(X)
  correct = 0
  svm_predicted = []
  svm_expected = []
  naivebayes_predicted = []
  naivebayes_expected = []
  decisiontree_predicted = []
  decisiontree_expected = []
  for i in range(len(X)):
    front_X = X[:i]
    test_X = X[i]
    back_X = X[i+1:]
    
    front_y = y[:i]
    test_y = y[i]
    back_y = y[i+1:]

    training_X = front_X + back_X
    training_y = front_y + back_y
    training_X = np.asarray(training_X)
    training_y = np.asarray(training_y)

    clf = svm.SVC(kernel='linear')
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
  print "########################################################################"
  print

  # Compare with Baseline
  # Run with Baseline
  # Load X_baseline
  X_baseline = pickle.load(open(os.path.join(path_pickles, 'X_2nd_Tier_Baseline.pickle'), 'r'))
  X_baseline = np.asarray(X_baseline)
  # Load y
  y_baseline = pickle.load(open(os.path.join(path_pickles, 'y_2nd_Tier_Baseline.pickle'), 'r'))
  y_baseline = np.asarray(y_baseline)

  X = np.asarray(X)
  y = np.asarray(y)
  print X.shape
  print y.shape
  print X_baseline.shape
  print y_baseline.shape
  print len(y_baseline[y_baseline==1])
