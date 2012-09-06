execfile('Utils.py')
import Utils
from sklearn import svm
import sys

if __name__ == "__main__":
  numOfInstances = 25
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.tools())
  # Pick a classifier model
  clf = svm.SVC()
  # Load DatasetTBA
  DatasetTBA = pickler.loadPickle(pickler.pathDatasetTBA)
  X = []
  for i in range(numOfInstances):
    d = DatasetTBA[i]
    X.append(d[2:])

  #X = []
  #X.append(DatasetTBA[0][2:])
  #X.append(DatasetTBA[2][2:])
  # Load Annotated Dataset / Target
  Target = pickler.loadPickle(pickler.pathTarget)
  y = []
  for i in range(numOfInstances):
    t = Target[i]
    if t == "-":
      # General
      y.append(0)
    else:
      y.append(1)

  #y = []
  #y.append(0)
  #y.append(1)

  #for i in range(numOfInstances):
  #  print str(X[i])+","+str(y[i])
  # Fit the classifier to get the Model
  Model = dataset_tools.prepModel(clf, X, y)
  pickler.dumpPickle(Model, "Model")
