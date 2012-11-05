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
  X = pickle.load(open(os.path.join(path_pickles, 'X_1st_Tier_Best.pickle'), 'r'))
  X = np.asarray(X)
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_1st_Tier_Best.pickle'), 'r'))
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
    X_train.append(x[5:])
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
    X_train.append(x[0:4] + x[7:])
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
    X_train.append(x[0:7] + x[8:])
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
  # Remove fourth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[0:8] + x[9:])
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
  # Remove fifth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[0:9])
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
  # Only first feature
  X_train = []
  for x in X:
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
  print "1st : " + str(correct) + " / " + str(total) + " = " + str(accuracy)
  # Only second feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[4:7])
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
    X_train.append([x[7]])
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
  # Only fourth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append([x[8]])
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
  # Only fifth feature
  X_train = []
  for x in X:
    x = list(x)
    X_train.append(x[9:])
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

  # Perform K-fold validation
  # Separate into k subsamples. Test on 1, train on k-1
  k = 7
  # Put X and y together to shuffle them
  Xy = []
  for i in range(len(X)):
    Xy.append((list(X[i]), y[i]))
  print Xy
  random.shuffle(Xy)
  print  ">>>>>"
  print Xy
  sys.exit()
  cross_fold = []
  for i in range(len(X)/k):
    X_cross_test = X[i*k:(i+1)*k]
    X_cross_train = X[0:i*k] + X[(i+1)*k:]
    y_cross_test = y[i*k:(i+1)*k]
    y_cross_train = y[0:i*k] + y[(i+1)*k:]

    X_cross_test = np.asarray(X_cross_test)
    X_cross_train = np.asarray(X_cross_train)
    y_cross_test = np.asarray(y_cross_test)
    y_cross_train = np.asarray(y_cross_train)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_cross_train, y_cross_train)
    expected = y_cross_test
    predicted = clf.predict(X_cross_test)
    total = float(len(expected))
    correct = float(0)
    for i in range(int(total)):
      if predicted[i] == expected[i]:
        correct += 1
    print y_cross_train
    print "Expected: " +  str(expected)
    print "Predicted: " +  str(predicted)
    print str(correct) + "/" + str(total) + " = " + str(correct/total)
    cross_fold.append(correct/total)
  print sum(cross_fold)/len(cross_fold)
  
  sys.exit()
  print "Compare with Baseline"
  # Compare with Baseline
  # Run with Baseline
  # Load X_baseline
  X_baseline = pickle.load(open(os.path.join(path_pickles, 'X_1st_Tier_Baseline.pickle'), 'r'))
  X_baseline = np.asarray(X_baseline)
  # Load y
  y_baseline = pickle.load(open(os.path.join(path_pickles, 'y_1st_Tier_Baseline.pickle'), 'r'))
  y_baseline = np.asarray(y_baseline)

  X = np.asarray(X)
  y = np.asarray(y)
  # Setting up baseline to have same number as unskewed
  X_n = []
  y_n = []
  X_y = []
  y_y = []
  for i in range(y_baseline.shape[0]):
    temp_x = list(X_baseline[i])
    temp_y = int(y_baseline[i])
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
   
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)

  clf = svm.SVC(kernel='linear')
  clf.fit(X,y)
  expected = y
  predicted = clf.predict(X)
  print metrics.classification_report(expected, predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train,y_train)
  expected = y_train
  predicted = clf.predict(X_train)
  print metrics.classification_report(expected, predicted)
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
