import cPickle as pickle
import os, sys
import operator
import numpy as np
import pylab as pl
from sklearn import svm, metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import LeavePOut
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one

if __name__ == "__main__":
  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']
  labels = ['g', 'y', 'n', 'u']
  reverse_labels = {'g':0, 'y':1, 'n':2, 'u':3}

  # Load Big_X
  #X = pickle.load(open(os.path.join(path_pickles, 'Big_X.pickle'),'r'))
  #X_raw = pickle.load(open(os.path.join(path_pickles, 'Big_X_With_Publish_Year.pickle'),'r'))
  X_raw = pickle.load(open(os.path.join(path_pickles, 'Big_X_Body_Text_PlusPlus.pickle'),'r'))

  # Load y
  y_raw = pickle.load(open(os.path.join(path_pickles, 'Y.pickle'),'r'))
  y_info_raw = pickle.load(open(os.path.join(path_pickles, 'Y_Info.pickle'),'r'))

  # Filter X_raw, y_raw and y_info_raw
  X = []
  y = []
  y_info = []
  for i in range(y_raw.shape[0]):
    temp_x = X_raw[i]
    temp_y = y_raw[i]
    temp_y_info = y_info_raw[i]
    if labels[temp_y] == 'g':
      #y.append(temp_y)
      continue
    elif labels[temp_y] == 'y':
      y.append(temp_y)
    elif labels[temp_y] == 'n':
      #y.append(reverse_labels['g'])
      y.append(temp_y)
    elif labels[temp_y] == 'u':
      continue
    X.append(temp_x)
    y_info.append(temp_y_info)
  X = np.asarray(X)
  y = np.asarray(y)
  y_info = np.asarray(y_info)
  num_of_labelled_data_points = y.shape[0]

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)
  expected = y_test
  predicted = clf.predict(X_test)

  print
  print "Feature Ranking with Recursive Feature Elimination"
  selector = RFE(clf, 1, step=1)
  selector = selector.fit(X, y)
  #print selector.support_
  print selector.ranking_
  print

  ## Build a classification task using 3 informative features
  #X, y = make_classification(n_samples=y.shape[0], n_features=X.shape[1], n_informative=3,
  #        n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=1,
  #                random_state=0)

  ## Create the RFE object and compute a cross-validated score.
  #svc = svm.SVC(kernel="linear")
  #rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
  #        loss_func=zero_one)
  #rfecv.fit(X, y)

  #print "Optimal number of features : %d" % rfecv.n_features_

  ## Plot number of features VS. cross-validation scores
  #pl.figure()
  #pl.xlabel("Number of features selected")
  #pl.ylabel("Cross validation score (nb of misclassifications)")
  #pl.plot(xrange(1, len(rfecv.cv_scores_) + 1), rfecv.cv_scores_)
  #pl.show()

  print "X_train" + str(X_train.shape)
  print "X_test" + str(X_test.shape)
  print "y_train" + str(y_train.shape)
  print "y_test" + str(y_test.shape)

  products = ("y", "y_test", "predicted")
  quantity = ([len(y[y == 0]),len(y[y == 1]),len(y[y == 2]),len(y[y == 3])],
              [len(y_test[y_test == 0]),len(y_test[y_test == 1]),len(y_test[y_test == 2]),len(y_test[y_test == 3])],
              [len(predicted[predicted == 0]),len(predicted[predicted == 1]),len(predicted[predicted == 2]),len(predicted[predicted == 3])])
  fw = 12
  print
  print ''.join([s.center(fw) for s in \
                 ('', 'C(general)', 'C(yes)', 'C(no)', 'C(undetermined)')])
  for i in range(len(products)):
    line = [products[i]]
    q = quantity[i]
    for j in range(0,4):
      line.append(q[j])
    print ''.join([str(s).center(fw) for s in line])
  print

  print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
  print "Legend:"
  print "0 - General"
  print "1 - Specific (Yes)"
  print "2 - Specific (No)"
  print "3 - Undetermined"
  print

  print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
  print

  clf = svm.SVC(kernel='linear')
  n_samples = X.shape[0]
  cv = cross_validation.ShuffleSplit(n_samples, n_iterations=10, test_size=0.1, random_state=0)
  print cv
  results = cross_validation.cross_val_score(clf, X, y, cv=cv)
  print results
  print

#Parameters
#----------
#n: int
#Total number of elements
#
#k: int
#Number of folds
#
#indices: boolean, optional (default True)
#Return train/test split as arrays of indices, rather than a boolean
#mask array. Integer indices are required when dealing with sparse
#matrices, since those cannot be indexed by boolean masks.
#
#shuffle: boolean, optional
#whether to shuffle the data before splitting into batches
#
#random_state: int or RandomState
#Pseudo number generator state used for random sampling.

  #kf = KFold(len(y), 10, indices=False, shuffle=True)
  #print kf
  #for train, test in kf:
  #  print train, test
  #print

  #loo = LeaveOneOut(len(y))
  #print loo
  #for train, test in loo:
  #  print train, test
  #print

  #lpo = LeavePOut(len(y), 2)
  #print lpo
  #for train, test in lpo:
  #  print train, test
  #print
