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
  # Remove first feature (5 parts)
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
