execfile('Utils.py')
import Utils
from sklearn import svm
import sys

if __name__ == "__main__":
  numOfInstances = 1
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.tools())
  # Pick a classifier model
  clf = svm.SVC()
  # Load DatasetTBA
  DatasetTBA = pickler.loadPickle(pickler.pathDatasetTBA)
  X = []
  for d in DatasetTBA:
    X.append(d[2:])
  # Load Annotated Dataset / Target
  Target = pickler.loadPickle(pickler.pathTarget)
  y = []
  for t in Target:
    if t == "-":
      # General
      y.append(-1)
    else:
      y.append(1)

  #for i in range(numOfInstances):
  #  print str(X[i])+","+str(y[i])
  #sys.exit()
  # Fit the classifier to get the Model
  Model = dataset_tools.prepModel(clf, X, y, numOfInstances)
  pickler.dumpPickle(Model, "Model")
