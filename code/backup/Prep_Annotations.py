import Utils
pickler = Utils.pickler()

datasetTBA = pickler.loadPickle('/Users/lwheng/Downloads/fyp/DatasetTBA.pickle')
numOfDigits = len(str(len(datasetTBA)))
for i in range(len(datasetTBA)):
  r = datasetTBA[i]
  cite_key = str(r[0]['citing']+"==>"+r[0]['cited'])
  print "#" + str(i).rjust(numOfDigits, '0') + "\t" + cite_key + ","
