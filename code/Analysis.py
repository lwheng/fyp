import cPickle as pickle
import os, sys
import numpy as np
from sklearn import svm, metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import LeavePOut
from sklearn.feature_selection import RFE

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
  X_raw = pickle.load(open(os.path.join(path_pickles, 'Big_X_With_Publish_Year.pickle'),'r'))

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
      y.append(temp_y)
    elif labels[temp_y] == 'y':
      y.append(temp_y)
    elif labels[temp_y] == 'n':
      y.append(reverse_labels['g'])
    elif labels[temp_y] == 'u':
      continue
    X.append(temp_x)
    y_info.append(temp_y_info)
  X = np.asarray(X)
  y = np.asarray(y)
  y_info = np.asarray(y_info)
  num_of_labelled_data_points = y.shape[0]

  train = int(0.9 * num_of_labelled_data_points)
  X_train = X[0:train]
  X_test = X[train:]
  y_train = y[0:train]
  y_test = y[train:]

  clf = svm.SVC()
  clf.fit(X_train, y_train)
  expected = y_test
  predicted = clf.predict(X_test)
  print "X_train" + str(X_train.shape)
  print "X_test" + str(X_test.shape)
  print "y_train" + str(y_train.shape)
  print "y_test" + str(y_test.shape)
  print

  print "Classification report for classifier:"
  print clf

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
  print metrics.classification_report(expected, predicted)

  #print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
  #print "Legend:"
  #print "0 - General"
  #print "1 - Specific (Yes)"
  #print "2 - Specific (No)"
  #print "3 - Undetermined"
  #print

  print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
  print

  print "Comparing prediction with answers"
  print "Printing only wrong predictions:"

  for i in range(len(expected)):
    index = i+train
    e = expected[i]
    e = labels[e]
    p = int(predicted[i])
    p = labels[p]
    if (e!=p):
      print y_info[index][0] + "  " + y_info[index][1] + "  " + str(y_info[index][2]).rjust(3, '0') + ": Expected-> " + str(e) + "-" + str(p) + " <-Predicted"
