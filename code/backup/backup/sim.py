import sys
import os
import math
import string
import getopt
import myUtils
import heapq
from nltk.stem.wordnet import WordNetLemmatizer

# Lemmatizer
lm = WordNetLemmatizer()

# No. of documents in corpora
N = 8889

mainDirectory = "/Users/lwheng/Downloads/fyp"

# For vocab
# Why bother loading vocab file when it is
# already captured in df
# vocabList = []
# vocabFile = "vocab-(2012-03-15-11:44:27.805232).txt"

# For DF
dfDict = {}
dfFile = "dftable-(2012-03-15-11:47:16.581332).txt"

# For fragmenting
fragmentSize = 15
lineRanges = []

# For d1
d1Filename = ""
d1Lines = []  # In display mode
d1Dict = {}
d1Vector = []

# For d2
d2Filename = ""
d2Lines = []  # In display mode
d2DFDict = {}
d2TFDict = {}
d2Fragments = []
d2FragmentsToTest = []
fragmentDict = {}
fragmentVector = []

# Booleans
weightSwitch = False
interactive = False
tfLoaded = False
dfLoaded = False

# For results
top = []
interactiveResults = []
resultsDict = {}
rangeOfTop = []

# Where all TF files are found
tfLocation = "tfLemmatized"
tfDirectory = str(os.path.join(mainDirectory, tfLocation))
# Where the DFTable (general) is found
dfPath = str(os.path.join(mainDirectory, dfFile))
# Where the vocab file is found
# vocabPath = str(os.path.join(mainDirectory,vocabFile))

def loadVocab():
  print "-----\tLoading Vocab\t-----"
  global vocabList
  global vocabPath
  vocabList = []
  openvocab = open(vocabPath,"r")
  for l in openvocab:
    line = myUtils.removespecialcharacters(l)
    vocabList.append(line)
  openvocab.close()

def loadD1():
  print "-----\tLoading D1\t-----"
  global d1Filename
  global d1Lines
  d1Lines = []
  opend1 = open(d1Filename,"r")
  for l in opend1:
    line = myUtils.removespecialcharacters(l)
    if len(line) != 0:
      d1Lines.append(line)
  opend1.close()

def loadD2():
  print "-----\tLoading D2\t-----"
  global d2Filename
  global d2Lines
  d2Lines = []
  opend2 = open(d2Filename,"r")
  for l in opend2:
    line = myUtils.removespecialcharacters(l)
    # if len(line) != 0:
    #   d2Lines.append(line)
    d2Lines.append(line)
  opend2.close()

def loadDFTable():
  print "-----\tLoading DFTable\t-----"
  global dfPath
  global dfDict
  global dfLoaded
  dfDict = {}
  if not dfLoaded:
    opendf = open(dfPath,"r")
    for l in opendf:
      tokens = l.split()
      dfDict[tokens[0]] = myUtils.removespecialcharacters(tokens[1])
    opendf.close()
    dfLoaded = True

def loadTFTable():
  print "-----\tLoading TFTable\t-----"
  global d2Filename
  global d2TFDict
  global tfDirectory
  global tfLoaded
  global weightSwitch
  d2TFDict = {}
  if not tfLoaded:
    tfFilename = ((d2Filename.split("/"))[-1]).replace(".txt", ".tf")
    tfPath = os.path.join(tfDirectory,tfFilename)
    if os.path.exists(tfPath):
      opentf = open(tfPath,"r")
      for l in opentf:
        tokens = l.split()
        term = tokens[0]
        locations = (myUtils.removespecialcharacters(tokens[1])).split("!")
        d2TFDict[term] = locations
      opentf.close()
      tfLoaded = True
    else:
      print "TF file " + tfFilename + " not found."
      print "Reverting back to weightless mode."
      weightSwitch = False
      tfLoaded = False

def prepD1(d1LinesInput):
  global d1Dict
  global weightSwitch
  d1Dict = {}
  for l in d1LinesInput:
    tokens = l.split()
    for t in tokens:
      toadd = ""
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        toadd = lm.lemmatize(toadd)
        if toadd not in d1Dict:
          d1Dict[toadd] = 0
        d1Dict[toadd] += 1
  if weightSwitch:
    for k in d1Dict:
      d1Dict[k] = d1Dict[k]*idf(k)

  global d1Vector
  global vocabList
  d1Vector = []
  for v in vocabList:
    if v in d1Dict:
      d1Vector.append(d1Dict[v])
    else:
      d1Vector.append(0)

def prepFragment(fragmentLinesInput,lineRangeInput):
  global fragmentDict
  global weightSwitch

  fragmentDict = {}
  for l in fragmentLinesInput:
    tokens = l.split()
    for t in tokens:
      toadd = ""
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        if toadd not in fragmentDict:
          fragmentDict[toadd] = 0
        fragmentDict[toadd] += 1

  if weightSwitch:
    for k in fragmentDict:
      if k in d2TFDict:
        locations = d2TFDict[k]
        tf = 0
        for location in locations:
          if int(location) in lineRangeInput:
            tf += 1
      # fragmentDict[k] = fragmentDict[k]*tf*idf(k)
      fragmentDict[k] = tf*idf(k)
      if fragmentDict[k] == 0:
        fragmentDict[k] = 0.0

  global vocabList
  global fragmentVector
  fragmentVector = []
  for v in vocabList:
    if v in fragmentDict:
      fragmentVector.append(fragmentDict[v])
    else:
      fragmentVector.append(0)

def prepd2Fragments(d2LinesInput):
  global fragmentSize
  global d2Fragments
  d2Fragments = []
  lineRangeTemp = []
  for i in xrange(0, len(d2LinesInput), fragmentSize):
    d2Fragments.append(d2LinesInput[i:i+fragmentSize])
    lineRangeTemp.append(range(i+1,i+1+fragmentSize))

  # Overlap the fragments
  global d2FragmentsToTest
  global lineRanges
  d2FragmentsToTest = []
  lineRanges = []
  if fragmentSize > 1:
    fragmentSizeHalf = fragmentSize/2
    d2FragmentsOverlap = []
    lineRangesOverlap = []
    for i in range(len(d2Fragments)-1):
      d2FragmentsOverlap.append(map(lambda x:x.lower(), d2Fragments[i]))
      d2FragmentsOverlap.append(map(lambda x:x.lower(),d2Fragments[i][fragmentSizeHalf:] + d2Fragments[i+1][0:fragmentSizeHalf]))
      lineRangesOverlap.append(lineRangeTemp[i])
      lineRangesOverlap.append(lineRangeTemp[i][fragmentSizeHalf:] + lineRangeTemp[i+1][0:fragmentSizeHalf])
    d2FragmentsOverlap.append(map(lambda x:x.lower(),d2Fragments[-1]))
    lineRangesOverlap.append(lineRangeTemp[-1])
    d2FragmentsToTest = d2FragmentsOverlap
    lineRanges = lineRangesOverlap
  else:
    d2FragmentsToTest = map(lambda x:map(lambda y:y.lower(), x),d2Lines)
    lineRanges = range(1, len(d2Lines)+1)

  # # Preparing the lineRanges using fragmentSize
  # global lineRanges
  # lineRanges = []
  # if fragmentSize > 1:
  #   fragmentSizeHalf = fragmentSize/2
  #   for i in range(len(d2Fragments)-1):
  #     lower = (i*fragmentSize)+1
  #     upper = (i+1)*fragmentSize
  #     lineRanges.append(range(lower,upper+1))
  #     lineRanges.append(range(lower+fragmentSizeHalf,upper+1+fragmentSizeHalf))
  #   lineRanges.append(range((len(d2Fragments)-1)*fragmentSize + 1, len(d2Fragments)*fragmentSize + 1))
  # else:
  #   for i in range(len(d2LinesInput)):
  #     lineRanges.append([i])


def loadFiles():
  print "-----\tLoading Files\t-----"
  # loadVocab()
  loadDFTable()
  loadD1()
  loadD2()
  if weightSwitch:
    # loadDFTable()
    loadTFTable()
  print "-----\tLoading Done!\t-----"

def log(x):
  return math.log(x)

def idf(term):
  global N
  global dfDict
  return (log(N) - log(int(dfDict[term])))

def cosinesim(v1,v2):
  # Computes cosine similarity
  # Both dotproduct and magnitude done simultaneously to improve efficiency
  if (len(v1) != len(v2)):
    return False
  output = 0
  dot = 0
  sumofsquares1 = 0
  sumofsquares2 = 0
  for i in range(len(v1)):
    dot = dot + (v1[i]*v2[i])
    sumofsquares1 = sumofsquares1 + (v1[i])**2
    sumofsquares2 = sumofsquares2 + (v2[i])**2
  # if sumofsquares2 == 0:
  #   return False
  return dot/((math.sqrt(sumofsquares1))*(math.sqrt(sumofsquares2)))

def maxsim(d1, d2):
  print "-----\tComputing Max Sim()\t-----"
  prepD1(map(lambda x:x.lower(),d1))
  prepd2Fragments(map(lambda x:x.lower(),d2))

  # Let's compute the results
  global interactiveResults
  global resultsDict
  global d2FragmentsToTest
  global lineRanges
  global d1Dict
  interactiveResults = []
  resultsDict = {}
  scores = []
  fragmentsCount = len(d2FragmentsToTest)
  for i in range(fragmentsCount):
    result = sim(d1Dict, d2FragmentsToTest[i], lineRanges[i])
    interactiveResults.append(result)
    resultsDict[result] = i
    scores.append(result)
  print "-----\tMax Sim() Computed!\t-----"

  print "-----\tResults\t-----"
  # print "Total no. of fragments:\t" + str(fragmentsCount)
  # print "Fragment Scores (Top 10 Only):"

  global top
  global rangeOfTop
  top = []
  top = heapq.nlargest(10,scores)
  rangeOfTop = [lineRanges[resultsDict[top[0]]][0],lineRanges[resultsDict[top[0]]][-1]]
  # for i in range(len(top)):
  #   print "Fragment " + str(resultsDict[top[i]]) + "\t" + str(top[i])

  # if top[0] == 0:
  #   print "No fragments match!!"
  #   print "------------------------------"
  #   print "The search query did not match any of the fragments. Score is 0.0"
  # else:
  #   print "------------------------------"
  #   print "Fragment " + str(resultsDict[top[0]]) + " has the highest score of " + str(top[0])
  #   print "------------------------------"
  #   print "Contents of fragment " + str(resultsDict[top[0]]) + ":"
  #   print d2FragmentsToTest[resultsDict[top[0]]]
  #   print "------------------------------"
  #   print "Location of fragment in domain document:"
  #   print "This fragment is found from line " + str(lineRanges[resultsDict[top[0]]][0]) + "-" + str(lineRanges[resultsDict[top[0]]][-1]) + " of the domain document"
  #   print "------------------------------"


def sim(d1,fragment,lineRange):
  # print d1
  # print fragment
  # print lineRange
  # print "#################################################################"
  prepFragment(fragment,lineRange)
  global d1Vector
  global fragmentVector
  cossim = cosinesim(d1Vector,fragmentVector)
  return cossim

def usage():
  print "USAGE: python " + sys.argv[0] + " [-i] [-w] [-n <fragment size>] -1 <d1file> -2 <d2file>"
  print "-i: To switch on interactive mode."
  print "-w: To switch on with TF-IDF-weight mode. Default is False"
  print "-n: To specify size of fragments (by no. of lines). Default is 5"
  print "<d1file> is the query file"
  print "<d2file> is the domain file"
  print "E.g. python " + sys.argv[0] + " -i -w -n 20 -1 search.txt -2 A00-1001.txt"

def main(argv):
  try:
    opts, args = getopt.getopt(argv, "iwn:1:2:")
    for opt, args in opts:
      if opt == "-1":
        global d1Filename
        d1Filename = args
      elif opt == "-2":
        global d2Filename
        d2Filename = args
      elif opt == "-w":
        global weightSwitch
        weightSwitch = True
      elif opt == "-n":
        global fragmentSize
        fragmentSize = int(args)
      elif opt == "-i":
        global interactive
        interactive = True
  except getopt.GetoptError:
    usage()
    sys.exit(2)

import cmd
class interactiveMode(cmd.Cmd):
  def do_df(self, term):
    if term:
      global dfDict
      if term in dfDict:
        print dfDict[term]
      else:
        print "Record not found!"
    else:
      print "USAGE: df [term]"
  def help_df(self):
    print "\n".join(["USAGE:\tdf [term]","OUTPUT:\tRetrieve document frequency for [term]."])

  def do_idf(self, term):
    if term:
      global dfDict
      if term in dfDict:
        print (log(N) - log(int(dfDict[term])))
      else:
        print "Record not found!"
    else:
      print "USAGE: idf [term]"
  def help_idf(self):
    print "\n".join(["USAGE:\tidf [term]","OUTPUT:\tRetrieve inverse document frequency for [term]."])

  def do_tf(self, term):
    global tfLoaded
    if term:
      if tfLoaded:
        global d2TFDict
        if term in d2TFDict:
          print len(d2TFDict[term])
          print "Lines: " + str(d2TFDict[term])
        else:
          print "Record not found!"
      else:
        print "TF file was not loaded!"
    else:
      print "USAGE:\ttf [term]"
  def help_tf(self):
    print "\n".join(["USAGE:\ttf [term]","OUTPUT:\tRetrieve term frequency for [term] in domain file."])

  def do_score(self, term):
    global interactiveResults
    global resultsDict
    global top
    if term:
      if term.isdigit() and (int(term) in range(0, len(interactiveResults))):
        print "Fragment " + str(term) + ": " + str(interactiveResults[int(term)]) 
      elif term == "max":
        print "Fragment " + str(resultsDict[top[0]]) + ":\t" + str(top[0])
    else:
      for i in range(len(interactiveResults)):
        print "Fragment " + str(i) + ": " + str(interactiveResults[i])
  def help_score(self):
    print "\n".join(["USAGE:\tscore [ | <fragment id> | max]","OUTPUT:\tPrints out the scores of the fragments."])

  def do_print(self, line):
    if line:
      tokens = line.split()
      if len(tokens)>2:
        print "USAGE:\tprint [-q | -f <fragment id> | -l <[<line no.>|<line range>]>]"
      else:
        try:
          opts, args = getopt.getopt(tokens, "qf:l:")
          for opt, args in opts:
            if opt == "-q":
              global d1Lines
              for l in d1Lines:
                print l
            elif opt == "-f":
              global d2FragmentsToTest
              index = int(args)
              if index>=0 and index<len(d2FragmentsToTest):
                printout = d2FragmentsToTest[index]
                for i in printout:
                  print i
              else:
                print "Out of range. Fragment ID range: 0-" + str(len(d2FragmentsToTest)) + "."
            elif opt == "-l":
              global d2Lines
              if "-" in args:
                tokens = args.split("-")
                start = int(tokens[0])
                end = int(tokens[1])
                if (start>0 and start<len(d2Lines)) and (end>0 and end<len(d2Lines)):
                  for i in range(start, end+1):
                    print d2Lines[i-1]
                else:
                  print "Out of range. Line no. range: 1-" + str(len(d2Lines)) + "."
              else:
                index = int(args)
                if index>0 and index<len(d2Lines):
                  print d2Lines[index-1]
                else:
                  print "Line no. range: 1-" + str(len(d2Lines))
        except getopt.GetoptError:
          print "USAGE:\tprint [-q | -f <fragment id> | -l <[<line no.>|<line range>]>]"
    else:
      print "USAGE:\tprint [-q | -f <fragment id> | -l <[<line no.>|<line range>]>]"
  def help_print(self):
    print "\n".join(["USAGE:\tprint [-q | -f <fragment id> | -l <[<line no.>|<line range>]>]","OUTPUT:\tPrints out the contents of query, fragment or lines."])

  def do_settings(self,line):
    global fragmentSize
    global weightSwitch
    global d2Lines
    if line:
      try:
        opts,args = getopt.getopt(line.split(), "wn:")
        for opt, args in opts:
          if opt == "-w":
            weightSwitch = not weightSwitch
            if weightSwitch:
              print "TF-IDF weight usage is now ON."
            else:
              print "TF-IDF weight usage is now OFF."
          elif opt == "-n":
            newsize = int(args)
            if newsize>0 and newsize<=len(d2Lines):
              fragmentSize = newsize
              print "Fragment size is now " + str(fragmentSize) + " lines."
            else:
              print "Invalid fragment size. Document has " + str(len(d2Lines)) + " lines."
      except getopt.GetoptError:
        print "USAGE:\tsettings [-w] [-n <fragment size>]"
    else:
      if weightSwitch:
        print "TF-IDF weight usage is now ON."
      else:
        print "TF-IDF weight usage is now OFF."
      print "Fragment size is now " + str(fragmentSize) + " lines."
  def help_settings(self):
    print "\n".join(["USAGE:\tsettings [-w] [-n <fragment size>]","OUTPUT:\tPrints current settings. Add respective tag to modify settings."])

  def do_query(self, line):
    if line:
      global fragmentSize
      global weightSwitch
      global tfLoaded
      global dfLoaded
      print "Current settings:"
      if weightSwitch:
        print "TF-IDF weight usage is now ON."
      else:
        print "TF-IDF weight usage is now OFF."
      print "Fragment Size:\t" + str(fragmentSize) + " lines."
      global d1Lines
      d1Lines = []
      d1Lines.append(line)
      if weightSwitch and (not tfLoaded) and (not dfLoaded):
        loadDFTable()
        loadTFTable()
      maxsim(d1Lines,d2Lines)
    else:
      print "USAGE:\tquery <your search query>"
      print "OUTPUT:\tRun computations using your new query."
      print "INFO:\tUse 'settings' command to modify settings."
  def help_query(self):
    print "\n".join(["USAGE:\tquery <your search query>","OUTPUT:\tComputes the cosine similarity between your query and the document.","INFO:\tUse 'settings' command to modify settings."])

  def do_quit(self, line):
    """INFO:\tExits the program."""
    print "Good Bye!"
    sys.exit()

  def do_EOF(self, line):
    """INFO:\tTerminates the program."""
    print 
    print "Good Bye!"
    sys.exit()

  def postloop(self):
    print

if __name__ == '__main__':
  if len(sys.argv) < 5:
    usage()
    sys.exit()
  main(sys.argv[1:])
  loadFiles()
  maxsim(d1Lines,d2Lines)
  generalResult = sim(d1Dict, map(lambda x:x.lower(),d2Lines),range(1,len(d2Lines)+1))
  print str(top[0]) + "!" + str(rangeOfTop[0]) + "-" + str(rangeOfTop[-1]) + "!" + str(generalResult)
  if interactive:
    print "-----\tEntering interactive mode\t-----"
    interactiveMode().cmdloop()
  sys.exit()



