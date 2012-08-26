# Precompile files
execfile('Classifier.py')
execfile('Utils.py')
execfile('Citprov.py')

import Citprov
import Classifier
import cPickle as pickle
import sys

run = Citprov.citprov()
cls = Classifier.classifier()

def prepSourceAndData(dataFile="/Users/lwheng/Dropbox/fyp/annotation/run01.txt"):
  source = []
  data = []
  for l in open(dataFile,'r'):
    cite_key = l.strip().split(',')[0]
    out = run.citProv(cite_key)
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
  run = Citprov.citprov()
  cls = Classifier.classifier()
  if len(sys.argv) >= 3:
    (source,data) = prepSourceAndData(sys.argv[1])
    target = prepTarget(sys.argv[2])
  else:
    (source, data) = prepSourceAndData()
    target = prepTarget()

  # Stop here if you don't have enough annotated target
  for s in source:
    print s
  sys.exit()
  cls.prepClassifier(data, target)
  for i in range(len(data)):
    print str(source[i][0]) + "\t" + "\t Prediction: " + str(cls.predict(data[i])) + "\t Annotation: " + str(target[i])
