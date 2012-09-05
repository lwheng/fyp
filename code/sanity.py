import Utils
pickler = Utils.pickler()

datasetTBA = pickler.loadPickle('/Users/lwheng/Downloads/fyp/DatasetTBA.pickle')
for r in datasetTBA:
  print str(r[0]['citing']+"==>"+r[0]['cited'])
  print str(r[1].attributes['citStr'].value)
  print pickler.authors[r[0]['cited']]
  print
