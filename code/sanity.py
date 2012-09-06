execfile('Utils.py')
import Utils

if __name__ == '__main__':
  numOfInstances = 1
  pickler = Utils.pickler()
  model = pickler.loadPickle(pickler.pathModel)
  dataset = pickler.loadPickle(pickler.pathDatasetTBA)
  observations = []
  for i in range(numOfInstances):
    observations.append(dataset[i][2:])
  for o in observations:
    print model.predict(o)
