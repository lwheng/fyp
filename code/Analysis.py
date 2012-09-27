import cPickle as pickle
import os, sys
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
  labels = ['g', 'y', 'n', 'u']

  # Load Big_X
  #X = pickle.load(open(os.path.join(path_pickles, 'Big_X.pickle'),'r'))
  X = pickle.load(open(os.path.join(path_pickles, 'Big_X_With_Publish_Year.pickle'),'r'))

  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'Y.pickle'),'r'))
  num_of_labelled_data_points = y.shape[0]

  # Set size of X to size of y
  X = X[0:num_of_labelled_data_points]

  train = int(0.9 * num_of_labelled_data_points)
  X_train = X[0:train]
  X_test = X[train:]
  y_train = y[0:train]
  y_test = y[train:]

  clf = svm.SVC()
  clf.fit(X_train, y_train)
  expected = y_test
  predicted = clf.predict(X_test)
  print clf
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

  print "Comparing prediction with answers"
  print "Printing only wrong predictions:"
  info = {}
  info[labels[0]+"-"+labels[1]] = []
  info[labels[0]+"-"+labels[2]] = []
  info[labels[0]+"-"+labels[3]] = []
  info[labels[1]+"-"+labels[0]] = []
  info[labels[1]+"-"+labels[2]] = []
  info[labels[1]+"-"+labels[3]] = []
  info[labels[2]+"-"+labels[0]] = []
  info[labels[2]+"-"+labels[1]] = []
  info[labels[2]+"-"+labels[3]] = []
  info[labels[3]+"-"+labels[0]] = []
  info[labels[3]+"-"+labels[1]] = []
  info[labels[3]+"-"+labels[2]] = []
  for i in xrange(train, len(y)):
    if (y[i] != int(predicted[i-train])):
      info[labels[y[i]] + "-" + labels[int(predicted[i-train])]].append(i)
      print "#" + str(i) + ": Answers = " + labels[y[i]] + " Prediction = " + labels[int(predicted[i-train])] + " Correct? " + str(y[i] == int(predicted[i-train]))
  for k in info.keys():
    print "Count(" + k +") = " + str(len(info[k]))
