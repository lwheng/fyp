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
  # Load Model_2nd_Tier
  model_2nd_tier = pickle.load(open(os.path.join(path_pickles, 'Model_2nd_Tier.pickle'), 'r'))
  print model_2nd_tier
