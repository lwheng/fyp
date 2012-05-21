import sys
import os
import math
import string
import getopt
import myUtils
import heapq
from nltk.stem.wordnet import WordNetLemmatizer

lm = WordNetLemmatizer()

N = 8889

D1Path = ""
D1Dict = {}
D1Vector = []
D1Lines = []

D2Path = ""
D2Dict = {}
D2Vector = []
D2Lines = []
D2TFDict = {}

DFPath = "/Users/lwheng/Downloads/fyp/dftable-(2012-03-15-11:47:16.581332).txt"
TFDirectory = "/Users/lwheng/Downloads/fyp/tfLemmatized"
FileDirectory = "/Users/lwheng/Downloads/fyp/FileLemmatizedCleaned/"

FragmentSize = 10
FragmentDict = {}
FragmentLines = []
FragmentVector = []

VocabList = []
Results = []
ResultsLineRange = []
LineRanges = []

interactive = False
weightOn = False

def loadDF():
  global DFPath
  global DFDict
  global VocabList
  DFDict = {}
  VocabList = []
  opendf = open(DFPath,"r")
  for l in opendf:
    tokens = l.split()
    DFDict[tokens[0]] = myUtils.removespecialcharacters(tokens[1])
    VocabList.append(tokens[0])
  opendf.close()

def loadD1():
  global lm
  global D1Path
  global D1Dict
  global D1Vector
  global D1Lines
  global weightOn
  D1Dict = {}
  D1Vector = []
  D1Lines = []
  opend1 = open(D1Path,"r")
  for l in opend1:
    line = myUtils.removespecialcharacters(l)
    D1Lines.append(line)
    tokens = line.split()
    for t in tokens:
      toadd = ""
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        toadd = lm.lemmatize(toadd)
        if toadd not in D1Dict:
          D1Dict[toadd] = 0
        D1Dict[toadd] += 1
  opend1.close()
  if weightOn:
    for k in D1Dict:
      D1Dict[k] = D1Dict[k] * idf(k)
  for v in VocabList:
    if v in D1Dict:
      D1Vector.append(D1Dict[v])
    else:
      D1Vector.append(0)

def prepD1(line):
  global lm
  global D1Path
  global D1Dict
  global D1Vector
  global D1Lines
  global weightOn
  D1Dict = {}
  D1Vector = []
  D1Lines = []
  D1Lines.append(line)
  tokens = line.split()
  for t in tokens:
    toadd = ""
    if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
      toadd = t
    elif (myUtils.removepunctuation(t)).isalnum():
      toadd = myUtils.removepunctuation(t)

    if len(toadd) != 0:
      toadd = lm.lemmatize(toadd)
      if toadd not in D1Dict:
        D1Dict[toadd] = 0
      D1Dict[toadd] += 1
  if weightOn:
    for k in D1Dict:
      D1Dict[k] = D1Dict[k] * idf(k)
  for v in VocabList:
    if v in D1Dict:
      D1Vector.append(D1Dict[v])
    else:
      D1Vector.append(0)

def loadD2():
  global D2Path
  global D2Dict
  global D2Vector
  global D2Lines
  global weightOn
  D2Dict = {}
  D2Vector = []
  D2Lines = []
  opend2 = open(D2Path,"r")
  for l in opend2:
    line = myUtils.removespecialcharacters(l)
    D2Lines.append(line)
    tokens = line.split()
    for t in tokens:
      toadd = ""
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        toadd = lm.lemmatize(toadd)
        if toadd not in D2Dict:
          D2Dict[toadd] = 0
        D2Dict[toadd] += 1
  opend2.close()
  if weightOn:
    for k in D2Dict:
      D2Dict[k] = D2Dict[k] * idf(k)

def loadTF():
  global D2Path
  global TFPath
  global TFDirectory
  global D2TFDict
  global weightOn
  D2TFDict = {}
  TFPath = os.path.join(TFDirectory, ((D2Path.split("/"))[-1]).replace(".txt",".tf"))
  opentf = open(TFPath,"r")
  for l in opentf:
    tokens = l.split()
    D2TFDict[tokens[0]] = (myUtils.removespecialcharacters(tokens[1])).split("!")
  opentf.close()

def loadFragment(lineRange):
  global D2Lines
  global FragmentLines
  global FragmentDict
  global FragmentVector
  FragmentLines = []
  FragmentDict = {}
  FragmentVector = []
  for i in lineRange:
    FragmentLines.append(D2Lines[i])
  for l in FragmentLines:
    tokens = l.split()
    for t in tokens:
      toadd = ""
      if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
        toadd = t
      elif (myUtils.removepunctuation(t)).isalnum():
        toadd = myUtils.removepunctuation(t)

      if len(toadd) != 0:
        if toadd not in FragmentDict:
          FragmentDict[toadd] = 0
        FragmentDict[toadd] += 1
  if weightOn:
    for k in FragmentDict:
      FragmentDict[k] = float(FragmentDict[k] * idf(k))
  for v in VocabList:
    if v in FragmentDict:
      FragmentVector.append(FragmentDict[v])
    else:
      FragmentVector.append(0)

def loadFiles():
  print "Loading files..."
  loadDF()
  loadD1()
  loadD2()
  loadTF()
  print "Loading done!"

def idf(term):
  global N
  global DFDict
  return (math.log(N) - math.log(int(DFDict[term])))

def sim(v1, v2):
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
  return dot/((math.sqrt(sumofsquares1))*(math.sqrt(sumofsquares2)))

def computeMaxSim():
  global D2Lines
  global LineRanges
  global FragmentSize

  LineRanges = []

  if FragmentSize > 1:
    FragmentSizeHalf = FragmentSize/2
    for i in xrange(0,len(D2Lines), FragmentSizeHalf):
      if (i+FragmentSize-1) < len(D2Lines):
        LineRanges.append(range(i, i+FragmentSize))
      else:
        LineRanges.append(range(i, len(D2Lines)))
  else:
    LineRanges = range(0, len(D2Lines))

  print "Computing..."
  global Results
  global ResultsLineRange
  global FragmentVector
  Results = []
  for lineRange in LineRanges:
    loadFragment(lineRange)
    Results.append(sim(D1Vector, FragmentVector))
    ResultsLineRange.append(lineRange)
  print "Computing Done!"

def printResults():
  global Results
  global ResultsLineRange
  top = list(Results)
  toprange = list(ResultsLineRange)
  for i in range(0,10):
    maxIndex = top.index(max(top))
    result = top.pop(maxIndex)
    resultrange = toprange.pop(maxIndex)
    print str(resultrange[0]) + "-" + str(resultrange[-1]) + "\t" + str(result)

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
        global D1Path
        D1Path = args
      elif opt == "-2":
        global D2Path
        D2Path = args
      elif opt == "-w":
        global weightOn
        weightOn = True
      elif opt == "-n":
        global FragmentSize
        FragmentSize = int(args)
      elif opt == "-i":
        global interactive
        interactive = True

    loadFiles()
    computeMaxSim()
    printResults()
  except getopt.GetoptError:
    usage()
    sys.exit(2)
  if interactive:
    interactiveMode().cmdloop()
  sys.exit()

import cmd
class interactiveMode(cmd.Cmd):
  def do_df(self, term):
    if term:
      global DFDict
      if term in DFDict:
        print DFDict[term]
      else:
        print "Record not found!"
    else:
      print "USAGE: df [term]"
  def help_df(self):
    print "\n".join(["USAGE:\tdf [term]","OUTPUT:\tRetrieve document frequency for [term]."])

  def do_idf(self, term):
    if term:
      global DFDict
      if term in DFDict:
        print idf(term)
      else:
        print "Record not found!"
    else:
      print "USAGE: idf [term]"
  def help_idf(self):
    print "\n".join(["USAGE:\tidf [term]","OUTPUT:\tRetrieve inverse document frequency for [term]."])

  def do_tf(self, term):
    if term:
      global D2TFDict
      if term in D2TFDict:
        print len(D2TFDict[term])
        print "Lines: " + str(D2TFDict[term])
      else:
        print "Record not found!"
    else:
      print "USAGE:\ttf [term]"
  def help_tf(self):
    print "\n".join(["USAGE:\ttf [term]","OUTPUT:\tRetrieve term frequency for [term] in domain file."])

  def do_score(self, term):
    printResults()
  def help_score(self):
    print "\n".join(["USAGE:\tscore","OUTPUT:\tPrints out the top scores of the fragments."])

  def do_print(self, line):
    if line:
      tokens = line.split()
      if len(tokens)>2:
        print "USAGE:\tprint [-q | -l <[<line no.>|<line range>]>]"
      else:
        try:
          opts, args = getopt.getopt(tokens, "ql:")
          for opt, args in opts:
            if opt == "-q":
              global D1Lines
              for l in D1Lines:
                print l
            elif opt == "-l":
              global D2Lines
              if "-" in args:
                tokens = args.split("-")
                start = int(tokens[0])
                end = int(tokens[1])
                if (start>=0 and start<len(D2Lines)) and (end>start and end<len(D2Lines)):
                  for i in range(start, end+1):
                    print D2Lines[i]
                else:
                  print "Out of range. Line no. range: 0-" + str(len(D2Lines)) + "."
              else:
                index = int(args)
                if index>=0 and index<len(D2Lines):
                  print D2Lines[index]
                else:
                  print "Line no. range: 0-" + str(len(D2Lines))
        except getopt.GetoptError:
          print "USAGE:\tprint [-q | -l <[<line no.>|<line range>]>]"
    else:
      print "USAGE:\tprint [-q | -l <[<line no.>|<line range>]>]"
  def help_print(self):
    print "\n".join(["USAGE:\tprint [-q | -l <[<line no.>|<line range>]>]","OUTPUT:\tPrints out the contents of query, fragment or lines."])

  def do_settings(self,line):
    global FragmentSize
    global weightOn
    global D2Path
    global D2Lines
    if line:
      try:
        opts,args = getopt.getopt(line.split(), "wn:")
        for opt, args in opts:
          if opt == "-w":
            weightOn = not weightOn
            if weightSwitch:
              print "TF-IDF weight usage is now ON."
            else:
              print "TF-IDF weight usage is now OFF."
          elif opt == "-n":
            newsize = int(args)
            if newsize>0 and newsize<=len(d2Lines):
              FragmentSize = newsize
              print "Fragment size is now " + str(FragmentSize) + " lines."
            else:
              print "Invalid fragment size. Document has " + str(len(D2Lines)) + " lines."
      except getopt.GetoptError:
        print "USAGE:\tsettings [-w] [-n <fragment size>]"
    else:
      print "TF-IDF weight usage is now " + ("ON" if weightOn else "OFF") + "."
      print "Fragment size is now " + str(FragmentSize) + " lines."
      print "D2Path is now " + D2Path
  def help_settings(self):
    print "\n".join(["USAGE:\tsettings [-w] [-n <fragment size>]","OUTPUT:\tPrints current settings. Add respective tag to modify settings."])

  def do_load(self, line):
    if line:
      global D2Path
      global FileDirectory
      temp = os.path.join(FileDirectory, line+".txt")
      if os.path.exists(temp):
        D2Path = temp
        loadD2()
        print line + " loaded!"
      else:
        print line + " not found!"
    else:
      print "\n".join(["USAGE:\tload <domain file>","INFO:\tLoads new D2 file."])
  def help_load(self):
    print "\n".join(["USAGE:\tload <domain file>","INFO:\tLoads new D2 file."])

  def do_query(self, line):
    if line:
      global FragmentSize
      global weightOn
      global D2Path
      print "Current settings:"
      print "Weight Switch:\t" + ("ON" if weightOn else "OFF") + "."
      print "Fragment Size:\t" + str(FragmentSize) + " lines."
      print "D2 Files:\t" + str(D2Path)
      prepD1(line)
      computeMaxSim()
      printResults()
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
  sys.exit()
