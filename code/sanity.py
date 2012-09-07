execfile('Utils.py')
import Utils
import sys

def makeReadable(prediction):
  if prediction == 0.0:
    return 'General'
  else:
    return 'Specific'

if __name__ == '__main__':
  numOfInstances = 4
  pickler = Utils.pickler()
  model = pickler.loadPickle(pickler.pathModel)
  X = pickler.loadPickle(pickler.pathDatasetTBA)
  y = pickler.loadPickle(pickler.pathTarget)
  observations = []
  for i in range(numOfInstances):
    o = X[i]
    print "Predicted " + str(makeReadable(model.predict(o)[0])) + " for Answer " + y[i]
