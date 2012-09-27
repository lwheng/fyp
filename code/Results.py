import cPickle as pickle
import os
from sklearn import svm, metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import LeavePOut
from sklearn.feature_selection import RFE

def interpret_prediction(output):
  arr = output[0]
  labels = ['g', 'y', 'n', 'u']
  max_proba = -1
  index = 0
  for i in range(len(arr)):
    num = arr[i]
    if num > max_proba:
      index = i
      max_proba = num
  return labels[index]

def interpret_label(label):
  labels = ['g', 'y', 'n', 'u']
  return labels[label]

if __name__ == "__main__":
  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Big_X
  #X = pickle.load(open(os.path.join(path_pickles, 'Big_X.pickle'),'r'))
  X = pickle.load(open(os.path.join(path_pickles, 'Big_X_With_Publish_Year.pickle'),'r'))

  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'Y.pickle'),'r'))
  num_of_labelled_data_points = y.shape[0]

  # Set size of X to size of y
  X = X[0:num_of_labelled_data_points]

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
  clf = svm.SVC(kernel="linear")
  clf.fit(X_train, y_train)
  expected = y_test
  predicted = clf.predict(X_test)
  print clf
  print
  #print "Feature Ranking with Recursive Feature Elimination"
  #selector = RFE(clf, 5, step=1)
  #selector = selector.fit(X, y)
  #print selector.support_
  #print selector.ranking_
  print
  print "X_train" + str(X_train.shape)
  print "X_test" + str(X_test.shape)
  print "y_train" + str(y_train.shape)
  print "y_test" + str(y_test.shape)
  print
  print "In y:"
  print "Count(general) = " + str(len(y[y == 0]))
  print "Count(yes) = " + str(len(y[y == 1]))
  print "Count(no) = " + str(len(y[y == 2]))
  print "Count(undetermined) = " + str(len(y[y == 3]))
  print
  print "In y_test:"
  print "Count(general) = " + str(len(y_test[y_test == 0]))
  print "Count(yes) = " + str(len(y_test[y_test == 1]))
  print "Count(no) = " + str(len(y_test[y_test == 2]))
  print "Count(undetermined) = " + str(len(y_test[y_test == 3]))
  print
  print "In predicted:"
  print "Count(general) = " + str(len(predicted[predicted == 0]))
  print "Count(yes) = " + str(len(predicted[predicted == 1]))
  print "Count(no) = " + str(len(predicted[predicted == 2]))
  print "Count(undetermined) = " + str(len(predicted[predicted == 3]))
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

  clf = svm.SVC(kernel='linear', C=1)
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
