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
	if weightOn:
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
		LineRanges = []
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
	top = Results
	for i in range(0,10):
		maxIndex = top.index(max(top))
		result = top.pop(maxIndex)
		resultrange = ResultsLineRange.pop(maxIndex)
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
		print "interactive mode"
	sys.exit()

# import cmd
# class interactiveMode(cmd.Cmd):
# 	print 

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	sys.exit()