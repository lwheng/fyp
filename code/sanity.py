execfile('Utils.py')
import Utils
import sys

def makeReadable(prediction):
  if prediction == 0.0:
    return 'General '
  else:
    return 'Specific'

if __name__ == '__main__':
  numOfInstances = 25
  pickler = Utils.pickler()
  model = pickler.loadPickle(pickler.pathModel)
  X = pickler.loadPickle(pickler.pathDatasetTBA)
  X_keys = pickler.loadPickle(pickler.pathDatasetTBA_keys)
  y = pickler.loadPickle(pickler.pathTarget)
  observations = []
  print "# of Data Points: " + str(numOfInstances)
  print model
  for i in range(numOfInstances):
    o = X[i]
    #print X_keys[i]['citing']+'==>'+X_keys[i]['cited'] + ": Predicted '" + str(makeReadable(model.predict(o)[0])) + "' for Answer " + y[i]
    print X_keys[i]['citing']+'==>'+X_keys[i]['cited'] + ": Predicted '" + str(model.predict_proba(o)[0]) + "' for Answer " + y[i]
  print 
