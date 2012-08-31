# Precompile files
execfile('Utils.py')
execfile('Citprov.py')

import Citprov
import Utils
import sys

def prepDataset(raw, experiment):
  dataset = []
  for e in experiment:
    out = run.citProv(e, raw[e['citing']+"==>"+e['cited']])
    dataset.extend(out)
  return dataset

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
  dataset_tools = Utils.dataset_tools(tools, dist)

  run = Citprov.citprov(nltk_Tools, tools, weight, dist, pickler)
  cls = Utils.classifier()
  experiment = dataset_tools.fetchExperiment(pickler.raw)
  dataset = prepDataset(pickler.raw, experiment)
  for d in dataset:
    print d
  print len(dataset)
  sys.exit()
  target = prepTarget()
  cls.prepClassifier(data[0:79], target)
  for i in range(len(data[0:79])):
    print str(source[i][0]) + "\t" + "\t Prediction: " + str(cls.predict(data[i])) + "\t Annotation: " + str(target[i])
