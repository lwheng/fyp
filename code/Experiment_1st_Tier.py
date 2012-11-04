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
  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'y_1st_Tier_Best.pickle'), 'r'))

  # Feature Ablation
  # Remove first feature (5 parts)
  X_train = []
  for x in X:
    X_train.append(x[5:])
  X_train.npasarray(X_train)
  print X.shape
  print X_train.shape
  sys.exit()
  clf = svm.SVC(kernel='linear')
  clf.fit(X, y)
  expected = y
  predicted = clf.predict(X)
