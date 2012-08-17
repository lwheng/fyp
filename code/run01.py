import Citprov
import sys
# Trying for classifying General/Specific first
experiment = "/Users/lwheng/Dropbox/fyp/annotation/run01.txt"
start = open(experiment, 'r')
run = Citprov.citprov()

# Prep data
source = []
for l in start:
  info = l.split(',')
  cite_key = info[0].strip()
  out = run.citProv(cite_key)
  source.extend(out)
data = []
for d in source:
  data.append(d[2:])
start.close()
# Prep target
targetFile = '/Users/lwheng/Dropbox/fyp/annotation/annotations50experiment_run01.txt'
start = open(targetFile, 'r')
dataset = []
i = 0
for l in start:
  temp = data[i]
  annotation = l.strip().split(',')[1]
  if annotation == '!':
    temp.append(1)
  else:
    temp.append(0)
  dataset.append(temp)
  i += 1
start.close()

print data
