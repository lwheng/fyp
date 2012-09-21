import cPickle as pickle
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
  # Output

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Big_X
  big_X = pickle.load(open(os.path.join(path_pickles, 'Big_X.pickle'),'r'))

  # Load y
  y = pickle.load(open(os.path.join(path_pickles, 'Y.pickle'),'r'))

  # Choose Classifier Model
  clf = svm.SVC(kernel='linear',probability=True)
  #clf = LogisticRegression()

  clf.fit(big_X, y)

  # Dump pickle
  pickle.dump(clf)
