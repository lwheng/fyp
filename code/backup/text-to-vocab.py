# This script takes in a text file and generates
# a Vacab file

# Input: A text file
# Output: A Vacab file

import os
import sys
import string
import getopt
import myUtils
from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

inputFile = ""
vocabFile = "/Users/lwheng/Downloads/fyp/text-to-vocab-(" + date + "-" + time + ").txt"

def vocab():
  global vocabFile
  global inputFile

  vocabDict = {}
  openthefile = open(inputFile, "r")
  for l in openthefile:
      line = myUtils.removespecialcharacters(l)
      line = line.lower()
      tokens = line.split()
      for t in tokens:
          toadd = ""
          if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
              toadd = t
          elif (myUtils.removepunctuation(t)).isalnum():
              toadd = myUtils.removepunctuation(t)

          if len(toadd) != 0:
              if toadd not in vocabDict:
                  vocabDict[toadd] = 0
              vocabDict[toadd] += 1
  openvocabfile = open(vocabFile, "w")
  for k in vocabDict:
      towrite = k + "\n"
      openvocabfile.write(towrite)

def usage():
  print "USAGE: python " + sys.argv[0] + " <input text file>"
  print "Default output location is Downloads, text-to-vocab-(timestamp).txt"
  print "To specify output, add this: -o <output filename>"

def main(argv):
  global inputFile
  inputFile = argv[0]
  try:
    opts, args = getopt.getopt(argv, "o:")
    for opt, args in opts:
        if opt == "-o":
            global vocabFile
            vocabFile = args
  except getopt.GetoptError:
    usage()
    sys.exit(2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit()
    main(sys.argv[1:])
    vocab()
