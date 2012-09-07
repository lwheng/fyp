execfile('Utils.py')
import Utils
from sklearn import svm
import sys

if __name__ == "__main__":
  numOfInstances = 4
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.nltk_tools(), Utils.tools())
  # Pick a classifier model
  clf = svm.SVC()
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
    else:
      temp_y.append(1)

  #for i in range(numOfInstances):
  #  print str(X[i])+","+str(y[i])
  # Fit the classifier to get the Model
  Model = dataset_tools.prepModel(clf, temp_X, temp_y)
  pickler.dumpPickle(Model, "Model")
