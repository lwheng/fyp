execfile("Utils.py")
import Utils
import cPickle as pickle
import os
import sys
import numpy as np
import random
from sklearn import svm, metrics
from sklearn import cross_validation

if __name__ == "__main__":
  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Model_2nd_Tier
  model_1st_tier = pickle.load(open(os.path.join(path_pickles, 'Model_1st_Tier.pickle'), 'r'))
  # Load X
  X = pickle.load(open(os.path.join(path_pickles, 'X_1st_Tier.pickle'), 'r'))
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_1st_Tier.pickle'), 'r'))

  # Prediction
  X_g = []
  y_g = []
  X_s = []
  y_s = []
  for i in range(y.shape[0]):
    temp_x = list(X[i])
    temp_y = int(y[i])
    if temp_y == 1:
      X_s.append(temp_x)
      y_s.append(temp_y)
    else:
      X_g.append(temp_x)
      y_g.append(temp_y)
  X_train = X_s
  y_train = y_s
  
  # Put X_g and y_g together
  Xy_g = []
  for i in range(len(X_g)):
    Xy_g.append((X_g[i], y_g[i]))

  # Pick randomly from Xy_g
  times = 2
  sample = random.sample(Xy_g, int(len(y_train)*times))
  for (d, a) in sample:
    X_train.append(d)
    y_train.append(a)
  
  # Perform leave-one-out evaluation
  total = len(X_train)
  correct = 0
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
    if predicted == expected:
      correct += 1
    #print "Predicted: " + str(predicted) + " " + str(expected) + " :Expected"
  print "Results: " + str(correct) + "/" + str(total) + " = " + str(float(correct) / float(total))
  
  sys.exit()
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)
  print X_train.shape
  print y_train
  #clf = svm.SVC(kernel='linear')
  clf = svm.SVC()
  #clf.fit(X, y)
  #predicted = clf.predict(X_train)
  #expected = y_train
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_train)
  expected = y_train
  print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
  print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
