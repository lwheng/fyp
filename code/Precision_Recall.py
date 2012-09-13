execfile('Utils.py')
import Utils
import random
import pylab as pl
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import sys

if __name__ == "__main__":
  numOfInstances = 168
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.nltk_tools(), Utils.tools())
  # Pick a classifier model
  clf = svm.SVC(kernel='linear',probability=True)
  #clf = LogisticRegression()
  # Load DatasetTBA
  X = pickler.loadPickle(pickler.pathDatasetTBA)
  temp_X = X[:numOfInstances]

  # Load Annotated Dataset / Target
  y = pickler.loadPickle(pickler.pathTarget)
  
  temp_y = []
  for i in range(numOfInstances):
    t = y[i]
    if t == "-":
      # General
      temp_y.append(0)
    elif t == '?':
      temp_y.append(0)
    else:
      temp_y.append(1)

  X = temp_X
  y = np.asarray(temp_y)

  #X, y = X[y != 2], y[y != 2]  # Keep also 2 classes (0 and 1)
  n_samples, n_features = X.shape
  p = range(n_samples)  # Shuffle samples
  random.seed(0)
  random.shuffle(p)
  X, y = X[p], y[p]
  half = int(n_samples / 2)

# Add noisy features
np.random.seed(0)
#X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]

# Run classifier
classifier = svm.SVC(kernel='linear', probability=True)
print classifier
probas_ = classifier.fit(X[:half], y[:half]).predict_proba(X[half:])

# Compute confusion matrix
cm_prediction_ = classifier.fit(X[:half], y[:half]).predict(X[half:])
cm = confusion_matrix(y[half:], cm_prediction_)

# Compute Precision-Recall and plot curve
precision, recall, thresholds = precision_recall_curve(y[half:], probas_[:, 1])
area = auc(recall, precision)
print "Area Under Curve: %0.2f" % area

pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example: AUC=%0.2f' % area)
pl.legend(loc="lower left")
pl.show()

print cm

# Show confusion matrix
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()
