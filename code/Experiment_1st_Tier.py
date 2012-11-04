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
  clf = svm.SVC(kernel='linear')
  clf.fit(X, y)
  expected = y
  predicted = clf.predict(X)
  correct=0
  total=predicted.shape[0]
  for i in range(total):
    if predicted[i] == expected[i]:
      correct += 1
  print correct
  print total
