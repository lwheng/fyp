execfile('Utils.py')
import Utils
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import sys

if __name__ == "__main__":
  numOfInstances = 200
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.nltk_tools(), pickler, Utils.tools())
  # Pick a classifier model
  clf = svm.SVC(kernel='linear',probability=True)
  #clf = LogisticRegression()
  # Load DatasetTBA
  #X = pickler.loadPickle(pickler.pathDatasetTBA)
  X = pickler.loadPickle(pickler.pathDatasetTBACFS)
  temp_X = X[:numOfInstances]

  # Load Annotated Dataset / Target
  y = pickler.loadPickle(pickler.pathTarget)
  
  temp_y = []
  for i in range(numOfInstances):
    t = y[i]
    if t == "-":
      # General
      temp_y.append(1)
    elif t == '?':
      temp_y.append(1)
    else:
      temp_y.append(-1)

  Model = dataset_tools.prepModel(clf, temp_X, temp_y)
  #pickler.dumpPickle(Model, "Model")
  pickler.dumpPickle(Model, "ModelCFS")
