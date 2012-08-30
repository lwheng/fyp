# Precompile files
execfile('Utils.py')
execfile('Citprov.py')

import Citprov
import Utils
import sys

def prepSourceAndData(thisDataset, experiment):
  source = []
  data = []
  for e in experiment:
    out = run.citProv(e, thisDataset[e['citing']+"==>"+e['cited']])
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
  nltk_Tools = Utils.nltk_tools()
  tools = Utils.tools()
  weight = Utils.weight()
  dist = Utils.dist()
  pickler = Utils.pickler()
  dataset = Utils.dataset(tools, dist)

  run = Citprov.citprov(nltk_Tools, tools, weight, dist, pickler, dataset)
  cls = Utils.classifier()
  (source, data) = prepSourceAndData(pickler.dataset, dataset.fetchExperiment(pickler.dataset))
  target = prepTarget()
  cls.prepClassifier(data[0:79], target)
  for i in range(len(data[0:79])):
    print str(source[i][0]) + "\t" + "\t Prediction: " + str(cls.predict(data[i])) + "\t Annotation: " + str(target[i])
