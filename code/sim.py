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

def loadDFCitStr():
  global DFCitStrPath
  global DFCitStrDict
  DFDictStrDict = {}
  opendfcitstr = open(DFCitStrPath,"r")
  for l in opendfcitstr:
    tokens = l.split()
    DFCitStrDict[tokens[0]] = myUtils.removespecialcharacters(tokens[1])
  opendfcitstr.close()

def loadD1():
  global lm
  global D1Path
  global D1Dict
  global D1Vector
  global D1Lines
  global weightOn
  global version
  D1Dict = {}
  D1Vector = []
  D1Lines = []

  if version >= 3:
    from xml.dom.minidom import parseString
    import Levenshtein
    global citations
    global D2Path
    global metadataPath
    # To use Parscit's output
    D1xmlPath = D1Path.replace("txt", "xml")
    D1xmlPath = D1xmlPath.replace(".xml", "-parscit.xml")

    # Get title of D2
    tokens = D2Path.split("/")
    D2id = tokens[-1].replace(".txt", "")
    metadataFile = metadataPath + D2id[0:3] + ".xml"
    D2Title = ""
    try:
      openmetadata = open(metadataFile,"r")
      metadata = openmetadata.read()
      openmetadata.close()
      dom = parseString(metadata)
      for item in dom.getElementsByTagName("paper"):
        idAttr = item.getAttribute("id")
        if idAttr == D2id.split("-")[1]:
          D2Title = item.getElementsByTagName("title")[0].firstChild.data
          break
    except IOError as e:
      print "Error!"
      sys.exit(2)

    try:
      xmlfile = open(D1xmlPath,"r")
      data = xmlfile.read()
      xmlfile.close()
      dom = parseString(data)
      citationList = dom.getElementsByTagName("citationList")
      citations = citationList[0].getElementsByTagName("citation")

      # Using Levenshtein to predict which paper it is citing
      bestIndex = 0
      maxRatio = 0
      citation = citations[0]
      title = citation.getElementsByTagName("title")[0].firstChild.data
      ratio = Levenshtein.ratio(D2Title, title)
      if ratio > maxRatio:
        maxRatio = ratio
        bestIndex = 0
      for i in range(len(citations)):
        citation = citations[i]
        tags = ["note", "booktitle", "journal"]
        titleTag = citation.getElementsByTagName("title")
        index = 0
        while titleTag == []:
          titleTag = citation.getElementsByTagName(tags[index])
          index += 1

        title = titleTag[0].firstChild.data
        ratio = Levenshtein.ratio(D2Title, title)
        if ratio > maxRatio:
          maxRatio = ratio
          bestIndex = i
      citation = citations[bestIndex] # predicted citation

      # As of now, assume 1 context
      contextTag = citation.getElementsByTagName("context")
      if contextTag == []:
        print "No context -> no citation in body of paper"
        sys.exit()
      else:
        citStr = contextTag[0].getAttribute("citStr")
        contextOriginal = contextTag[0].firstChild.data
        context = contextOriginal.replace("("+citStr+")", "")
        line = context.lower()
        line = myUtils.removespecialcharacters(line)
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
            if version >= 4:
              D1Dict[k] = D1Dict[k] * idfD1(k)
            else:
              D1Dict[k] = D1Dict[k] * idf(k)
        for v in VocabList:
          if v in D1Dict:
            D1Vector.append(D1Dict[v])
          else:
            D1Vector.append(0)
    except IOError as e:
      print "Error!"
      sys.exit()

  else:
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
        #toadd = lm.lemmatize(toadd)
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

def loadInterlink():
  global interlinkPath
  global interlinkDict
  openthefile = open(interlinkPath,"r")
  for l in openthefile:
    tokens = l.replace(" ", "").replace("\n","").split("==>")
    if tokens[0] in interlinkDict:
      temp = interlinkDict[tokens[0]]
      temp.append(tokens[1])
      interlinkDict[tokens[0]] = temp
    else:
      temp = []
      temp.append(tokens[1])
      interlinkDict[tokens[0]] = temp
  openthefile.close()

def loadFiles():
  global version
  print "Loading files..."
  loadDF()
  if version >= 4:
    # This version use tf-idf for citing sentences
    loadDFCitStr()
  loadD1()
  loadD2()
  loadTF()
  print "Loading done!"

def idfD1(term):
  global N
  global DFDict
  global DFCitStrDict
  global version

  if version >= 4:
    return (math.log(N) - math.log(int(DFCitStrDict[term])))
  else:
    return (math.log(N) - math.log(int(DFDict[term])))

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
  length = 10 # to print only the top 10 results
  if len(top) < length:
    length = len(top)
  for i in range(length):
    maxIndex = top.index(max(top))
    result = top.pop(maxIndex)
    resultrange = toprange.pop(maxIndex)
    print str(resultrange[0]) + "-" + str(resultrange[-1]) + "\t" + str(result)

def checkRelation():
  # To check in interlink file whether citing relationship exists
  global D1Path
  global D2Path
  global interlinkDict
  tokens = D1Path.split("/")
  citing = tokens[-1].replace(".txt", "")
  tokens = D2Path.split("/")
  cited = tokens[-1].replace(".txt", "")
  # citing and cited are paper IDs, check against the interlink file
  if citing in interlinkDict:
    temp = interlinkDict[citing]
    if cited in temp:
      return True
    else:
      return False
  else:
    return False


def usage():
  print "USAGE: python " + sys.argv[0] + " [-v <version>] [-i] [-w] [-n <fragment size>] -1 <d1file> -2 <d2file>"
  print "-v: To switch versions. Default version is latest. -V to list versions."
  print "-i: To switch on interactive mode."
  print "-w: To switch on with TF-IDF-weight mode. Default is False"
  print "-n: To specify size of fragments (by no. of lines). Default is 5"
  print "<d1file> is the query file"
  print "<d2file> is the domain file"
  print "E.g. python " + sys.argv[0] + " -i -w -n 20 -1 search.txt -2 A00-1001.txt"

def main(argv):
  try:
    opts, args = getopt.getopt(argv, "v:Viwn:1:2:")
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
      elif opt == "-v":
        global version
        version = int(args)
      elif opt == "-V":
        global versions
        print "Versions: " + str(versions)
        sys.exit()

    print "Running on version: " + str(version)

    if version >= 2:
      loadInterlink()
      if not checkRelation():
        print "Citation relationship not found. D1 probably does not cite D2."
        sys.exit(2)
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
            if weightOn:
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
      if weigthOn:
        print "TF-IDF weight usage is now ON."
      else:
        print "TF-IDF weight usage is now OFF."
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
  import sys
  if len(sys.argv) == 1:
    usage()
    sys.exit()
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

  DFDict = {}
  DFCitStrDict = {}

  DFPath = "/Users/lwheng/Downloads/fyp/dftable-(2012-03-15-11:47:16.581332).txt"
  DFCitStrPath = "/Users/lwheng/Downloads/fyp/parscit-to-df-(2012-05-24-11:10:08.917366).txt" # For Version 4
  TFDirectory = "/Users/lwheng/Downloads/fyp/tfLemmatized"
  FileDirectory = "/Users/lwheng/Downloads/fyp/FileLemmatizedCleaned/"

  FragmentSize = 15 # Default size of fragments
  FragmentDict = {}
  FragmentLines = []
  FragmentVector = []

  metadataPath = "/Users/lwheng/Downloads/fyp/metadata/"

  interlinkPath = "/Users/lwheng/Dropbox/fyp/interlink/aan/acl.20080325.net"
  interlinkDict = {}

  VocabList = []
  Results = []
  ResultsLineRange = []
  LineRanges = []

  interactive = False
  weightOn = False

  versions = [1,2,3,4]
  version = versions[-1] # set default version to latest
  # Versions (Each point describes the version)
  # 1. Target file is broken down to fragments, search query in single text file. All TF & IDF generated from same vocab file
  # 2. Check whether citing relationship exists
  # 3. To properly use citing sentences for search query (modify loadD1). need to use *-parscit.xml. Exists cases with context -> confirm General?
  # 4. To have tf-idf for citing sentence
  main(sys.argv[1:])
  sys.exit()
