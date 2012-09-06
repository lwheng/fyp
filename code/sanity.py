execfile('Utils.py')
import Utils
import sys

if __name__ == '__main__':
  numOfInstances = 25
  pickler = Utils.pickler()
  model = pickler.loadPickle(pickler.pathModel)
  dataset = pickler.loadPickle(pickler.pathDatasetTBA)
  observations = []
  for i in range(numOfInstances):
    observations.append(dataset[i][2:])
  #print model.predict(dataset[0][2:])
  #print model.predict(dataset[2][2:])
  #sys.exit()
  for o in observations:
    print "Predicted " + str(model.predict(o)) + " for " + str(o)
