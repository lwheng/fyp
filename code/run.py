# Precompile files
execfile('Classifier.py')
execfile('Utils.py')
execfile('Citprov.py')

import Citprov
import Classifier
import Utils
import sys

def prepSourceAndData():
  source = []
  data = []
  experiment = dataset.fetchExperiment()
  for e in experiment:
    out = run.citProv(e)
    print out
    source.extend(out)
  for s in source:
    data.append(s[2:])
  return (source, data)

def prepTarget(targetFile="/Users/lwheng/Dropbox/fyp/annotation/annotations50experiment_run01.txt"):
  target = []
  for l in open(targetFile,'r'):
    annotation = l.strip().split(',')[1]
    if annotation == '!':
      target.append(0)
    else:
      target.append(1)
  return target

if __name__ == '__main__':
  pickler = Utils.pickler()
  tools = Utils.tools()
  dist = Utils.dist()
  weight = Utils.weight()
  dataset = Utils.dataset(tools, dist)
  nltk_Tools = Utils.nltk_tools()

  run = Citprov.citprov(pickler, tools, dist, weight, dataset, nltk_Tools)
  cls = Classifier.classifier()

  (source, data) = prepSourceAndData()
  target = prepTarget()

  # Stop here if you don't have enough annotated target
  sys.exit()
  cls.prepClassifier(data, target)
  for i in range(len(data)):
    print str(source[i][0]) + "\t" + "\t Prediction: " + str(cls.predict(data[i])) + "\t Annotation: " + str(target[i])
