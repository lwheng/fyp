import Citprov
import sys
# Trying for classifying General/Specific first
experiment = "/Users/lwheng/Dropbox/fyp/annotation/run01.txt"
start = open(experiment, 'r')
run = Citprov.citprov()
data = []
for l in start:
  info = l.split(',')
  cite_key = info[0].strip()
  out = run.citProv(cite_key)
  data.extend(out)
for i in range(len(data)):
  print data[i]
