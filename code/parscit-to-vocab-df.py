# This script takes in citingsentences.txt and generates a Vocab file, and a DFtable file

# Input: citingsentences.txt
# Output: A vocab file, a DFtable file

import os
import sys
import string
import getopt
import myUtils
from nltk.stem.wordnet import WordNetLemmatizer
from xml.dom.minidom import parseString
from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

inputFile = ""
vocabFile = "/Users/lwheng/Downloads/fyp/parscit-to-vocab-(" + date + "-" + time + ").txt"
dfFile = "/Users/lwheng/Downloads/fyp/parscit-to-df-(" + date + "-" + time +").txt"

lm = WordNetLemmatizer()

DFDict = {}
docList = []

def vocab():
  global inputFile
  global vocabFile
  global dfFile
  global DFDict

  opentheinput = open(inputFile, "r")
  index = 0
  for l in opentheinput:
    line = myUtils.removespecialcharacters(l)
    tokens = line.split("-parscit.xml:")
    paperID = tokens[0]
    contents = parseString(tokens[1])
    context = contents.getElementsByTagName("context")[0]
    citStr = context.getAttribute("citStr")
    citSentOriginal = context.firstChild.data
    citSent = citSentOriginal.replace("("+citStr+")", "")
    # citSent = citSent.lower()
    citSent = citSent.lower().encode("ascii", "ignore")
    tokens = citSent.split()

    if paperID not in docList:
      docList.append(paperID)
      newPaper = True
    else:
      newPaper = False

    tokenList = []
    for t in tokens:
      toadd = ""
      t = lm.lemmatize(t)
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        if toadd not in tokenList:
          tokenList.append(toadd)

    for item in tokenList:
      if item not in DFDict:
        DFDict[item] = 1
      else:
        if not newPaper:
          DFDict[item] += 1
  opentheinput.close()

  opendffile = open(dfFile, "w")
  openvocabfile = open(vocabFile, "w")
  for k in DFDict:
    writeVocab = k + "\n"
    writeDF = k + "\t" + str(DFDict[k]) + "\n"
    openvocabfile.write(writeVocab)
    opendffile.write(writeDF)
  opendffile.close()
  openvocabfile.close()

def usage():
  print "USAGE: python " + sys.argv[0] + " <input text file>"
  print "Default output location is Downloads, parscit-to-vocab-(timestamp).txt"
  print "To specify output, add this: -v <vocab output filename>, -d <df table output filename>"

def main(argv):
  global inputFile
  global vocabFile
  global dfFile
  inputFile = argv[0]
  try:
    opts, args = getopt.getopt(argv, "o:d:")
    for opt, args in opts:
      if opt == "-o":
        vocabFile = args
      elif opt == "-d":
        dfFile = args
  except getopt.GetoptError:
    usage()
    sys.exit(2)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    usage()
    sys.exit(2)
  main(sys.argv[1:])
  vocab()
