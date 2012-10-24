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
  model_2nd_tier = pickle.load(open(os.path.join(path_pickles, 'Model_2nd_Tier.pickle'), 'r'))
  # Load X
  X = pickle.load(open(os.path.join(path_pickles, 'X_2nd_Tier.pickle'), 'r'))
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_2nd_Tier.pickle'), 'r'))

  for i in range(len(X.shape[0])):
    print X[i] + " >>>>> " + y[i]
  print X.shape
  print y.shape

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
  
  # Pick randomly
  times = 2
  X_train.extend(random.sample(X_n, int(len(y_train)*times)))
  y_train.extend(random.sample(y_n, int(len(y_train)*times)))
  
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
