import Citprov
import Classifier
import cPickle as pickle
import sys

run = Citprov.citprov()
cls = Classifier.classifier()

experiment = "/Users/lwheng/Dropbox/fyp/annotation/run01.txt"
annotationFile = "/Users/lwheng/Dropbox/fyp/annotation/annotations50experiment_run01.txt"

# Prep data
source = []
data = []
for l in open(experiment,'r'):
  cite_key = l.strip().split(',')[0]
  out = run.citProv(cite_key)
  source.extend(out)
for s in source:
  data.append(s[2:])

print data
sys.exit()

# Prep target
target = []
for l in open(annotationFile,'r'):
  annotation = l.strip().split(',')[1]
  if annotation == '!':
    target.append(0)
  else:
    target.append(1)

cls.prepClassifier(data, target)
print cls.predict(data[0])
