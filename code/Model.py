execfile('Utils.py')
import Utils
from sklearn import svm

if __name__ == "__main__":
  numOfInstances = 26
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
    y.append(t[-1])
  # Fit the classifier to get the Model
  Model = dataset_tools.prepModel(clf, X, y, numOfInstances)
  pickler.dumpPickle(Model, "Model")
