import os
import sys
import cPickle as pickle

# To generate 500 records
# Requirements:
#   1. Has pdfbox file
#   2. Has parscit file
#   3. Has parscit section file
#   4. For citing, must have contexts

parscitSectionPath = "/Users/lwheng/Downloads/fyp/parscitsectionxml"
parscitPath = "/Users/lwheng/Downloads/fyp/parscitxml"
pdfboxPath = "/Users/lwheng/Downloads/fyp/pdfbox-0.72"

numOfRecords = 722
records = []

def checkFileInDirectory(path, filename, extension):
  fullpath = os.path.join(path, filename+extension)
  if os.path.exists(fullpath):
    return True
  else:
    return False

# Open the file that lists all records
annotationMasterFile = "/Users/lwheng/Dropbox/fyp/annotation/annotationsMaster.txt"
for l in open(annotationMasterFile,'r'):
  info = l.strip().replace(",","").split("==>")
  citing = info[0]
  cited = info[1]

  e = ['-parscit-section.xml', '-parscit.xml', '.txt']
  if checkFileInDirectory(parscitSectionPath, citing, e[0]):
    if checkFileInDirectory(parscitPath, citing, e[1]):
      if checkFileInDirectory(pdfboxPath+"/"+citing[0]+"/"+citing[0:3], citing, e[2]):
        if checkFileInDirectory(parscitSectionPath, cited, e[0]):
          if checkFileInDirectory(parscitPath, cited, e[1]):
            if checkFileInDirectory(pdfboxPath+"/"+cited[0]+"/"+cited[0:3], cited, e[2]):
              temp = {}
              temp['citing'] = citing
              temp['cited'] = cited
              records.append(temp)

pickle.dump(records[0:numOfRecords], open('Experiment.pickle','wb'))
