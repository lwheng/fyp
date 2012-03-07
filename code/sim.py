import sys
import os
import math
import string
import getopt
import myUtils
import heapq
# from heapq import nlargest

# No. of documents in corpora
N = 500

mainDirectory = "/Users/lwheng/Desktop"
dfFile = "dftable-(2012-03-07-20:25:49.094468).txt"
vocabFile = "vocab-(2012-03-07-20:17:59.878950).txt"

# Where all TF files are found
tfDirectory = str(os.path.join(mainDirectory, "tf"))
# Where the DFTable (general) is found
dfPath = str(os.path.join(mainDirectory, dfFile))
# Where the vocab file is found
vocabPath = str(os.path.join(mainDirectory,vocabFile))

vocabList = []          # Vocab
dfDict = {}             # DFTable
weightSwitch = False    # Weight switch, to use TF-IDF or not
fragmentSize = 5        # Default fragment size
d1Filename = ""         # Query file
d2Filename = ""         # Search domain file
d1Lines = []            # Lines of d1, for printing only
d2Lines = []            # Lines of d2, for printing only
d2DFTable = {}          # DFTable specifically for d2
d2TFDict = {}           # TFTable specifically for d2

# To capture info about d1
d1Dict = {}
d1Vector = []

# To capture fragments of d2
d2Fragments = []
d2FragmentsToTest = []

# To capture the results
top = []
interactiveResults = []
resultsDict = {}

def loadVocab():
	global vocabList
	global vocabPath
	openvocab = open(vocabPath,"r")
	for l in  openvocab:
		line = myUtils.removespecialcharacters(l)
		vocabList.append(line)
	openvocab.close()

def loadD1():
	global d1Filename
	global d1Lines
	opend1 = open(d1Filename,"r")
	for l in opend1:
		line = myUtils.removespecialcharacters(l)
		line = line.lower()
		if len(line) != 0:
			d1Lines.append(line)
	opend1.close()

def loadD2():
	global d2Filename
	global d2Lines
	opend2 = open(d2Filename,"r")
	for l in opend2:
		line = myUtils.removespecialcharacters(l)
		if len(line) != 0:
			d2Lines.append(line)
	opend2.close()

def loadDFTable():
	global dfPath
	global dfDict
	opendf = open(dfPath,"r")
	for l in opendf:
		tokens = l.split()
		dfDict[tokens[0]] = myUtils.removespecialcharacters(tokens[1])
	opendf.close()

def loadTFTable():
	global d2Filename
	global d2TFDict
	tokens = d2Filename.split("/")
	tfFilename = tokens[-1].replace(".txt", ".tf")
	tfPath = os.path.join(tfDirectory,tfFilename)
	if os.path.exists(tfPath):
		opentf = open(tfPath,"r")
		for l in opentf:
			tokens = l.split()
			term = tokens[0]
			locations = (myUtils.removespecialcharacters(tokens[1])).split("!")
			d2TFDict[term] = locations
		opentf.close()
	else:
		print "TF file " + tfFilename + " not found."
		print "Reverting back to weightless mode."
		global weightSwitch
		weightSwitch = False

def loadFiles():
	print "-----\tLoading Files\t-----"
	loadVocab()
	loadD1()
	loadD2()
	if weightSwitch:
		loadDFTable()
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
	return dot/((math.sqrt(sumofsquares1))*(math.sqrt(sumofsquares2)))

def maxsim(d1lines, d2lines):
	# Remember: d1lines and d2lines are only for printing
	print "-----\tComputing Max Sim()\t-----"

	# Lowercase all lines, for computation use
	d1 = []
	d2 = []
	for i in range(len(d1lines)):
		d1.append(d1lines[i].lower())
	for i in range(len(d2lines)):
		d2.append(d2lines[i].lower())

	# Divide D2 into fragments using fragmentSize
	# Note that d2Fragments has no overlapping
	global fragmentSize
	global d2Fragments
	d2Fragments = []
	for i in xrange(0, len(d2), fragmentSize):
		d2Fragments.append(d2[i:i+fragmentSize])

	# Overlap the fragments
	global d2FragmentsToTest
	d2FragmentsToTest = []
	if fragmentSize > 1:
		fragmentSizeHalf = fragmentSize/2
		d2FragmentsOverlap = []
		for i in range(len(d2Fragments)-1):
			d2FragmentsOverlap.append(d2Fragments[i])
			d2FragmentsOverlap.append(d2Fragments[i][fragmentSizeHalf:] + d2Fragments[i+1][0:fragmentSizeHalf])
		d2FragmentsOverlap.append(d2Fragments[-1])
		d2FragmentsToTest = d2FragmentsOverlap
	else:
		d2FragmentsToTest = d2Fragments

	# Preparing the lineRanges using fragmentSize
	lineRanges = []
	if fragmentSize > 1:
		fragmentSizeHalf = fragmentSize/2
		for i in range(len(d2Fragments)-1):
			lower = (i*fragmentSize)+1
			upper = (i+1)*fragmentSize
			lineRanges.append(range(lower,upper+1))
			lineRanges.append(range(lower+fragmentSizeHalf,upper+1+fragmentSizeHalf))
		lineRanges.append(range((len(d2Fragments)-1)*fragmentSize + 1, len(d2Fragments)*fragmentSize + 1))
	else:
		for i in range(len(d2Lines)):
			lineRanges.append([i])

	# In this version, V and DF is "general"
	# So we use this V to generate the vectors, and DF
	# to compute the TD.IDF value

	# We compute the d1Dict, since it is only to be computed one
	global d1Dict
	for l in d1:
		tokens = l.split()
		for t in tokens:
			toadd = ""
			if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
				toadd = t
			elif (myUtils.removepunctuation(t)).isalnum():
				toadd = myUtils.removepunctuation(t)

			if len(toadd) != 0:
				if toadd not in d1Dict:
					d1Dict[toadd] = 0
				d1Dict[toadd] += 1
	if weightSwitch:
		for k in d1Dict:
			# We don't need tf since tf = d1Dict[k]
			d1Dict[k] = d1Dict[k]*idf(k)
	# Now we generate the vector for d1
	global vocabList
	for v in vocabList:
		if v in d1Dict:
			d1Vector.append(d1Dict[v])
		else:
			d1Vector.append(0)

	# Let's compute the results
	global interactiveResults
	global resultsDict
	resultsDict = {}
	scores = []
	fragmentsCount = len(d2FragmentsToTest)
	for i in range(fragmentsCount):
		result = sim(None, d2FragmentsToTest[i], fragmentsCount, lineRanges[i])
		interactiveResults.append(result)
		resultsDict[result] = i
		scores.append(result)
	print "-----\tMax Sim() Computed!\t-----"

	print "-----\tResults\t-----"
	print "Total no. of fragments:\t" + str(fragmentsCount)
	print "Fragment Scores (Top 10 Only):"

	global top
	top = heapq.nlargest(10,scores)
	for i in range(len(top)):
		print "Fragment " + str(resultsDict[top[i]]) + "\t" + str(top[i])

	if top[0] == 0:
		print "No fragments match!!"
		print "------------------------------"
		print "The search query did not match any of the fragments. Score is 0.0"
	else:
		print "------------------------------"
		print "Fragment " + str(resultsDict[top[0]]) + " has the highest score of " + str(top[0])
		print "------------------------------"
		print "Contents of fragment " + str(resultsDict[top[0]]) + ":"
		print d2FragmentsToTest[resultsDict[top[0]]]
		print "------------------------------"
		print "Location of fragment in domain document:"
		print "This fragment is found from line " + str(lineRanges[resultsDict[top[0]]][0]) + "-" + str(lineRanges[resultsDict[top[0]]][-1]) + " of the domain document"
		print "------------------------------"


def sim(d1,fragment,fragmentsCount,lineRange):
	# d1 is query in lines
	# fragment is fragment in lines, aka the "d2"
	# fragmentsCount is no. of fragments

	d1dict = {}
	d1vector = []
	fragmentdict = {}
	fragmentvector = []
	
	if d1:
		for l in d1:
			tokens = l.split()
			for t in tokens:
				toadd = ""
				if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
					toadd = t
				elif (myUtils.removepunctuation(t)).isalnum():
					toadd = myUtils.removepunctuation(t)

				if len(toadd) != 0:
					if toadd not in d1dict:
						d1dict[toadd] = 0
					d1dict[toadd] += 1
	else:
		global d1Dict
		global d1Vector
		d1dict = d1Dict
		d1vector = d1Vector

	for l in fragment:
		tokens = l.split()
		for t in tokens:
			toadd = ""
			if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
				toadd = t
			elif (myUtils.removepunctuation(t)).isalnum():
				toadd = myUtils.removepunctuation(t)

			if len(toadd) != 0:
				if toadd not in fragmentdict:
					fragmentdict[toadd] = 0
				fragmentdict[toadd] += 1

	if weightSwitch:
		if d1:
			for k in d1dict:
				# We don't need tf since tf = d1dict[k]
				d1dict[k] = d1dict[k]*idf(k)
		for k in fragmentdict:
			if k in d2TFDict:
				locations = d2TFDict[k]
				tf = 0
				for location in locations:
					if int(location) in lineRange:
						tf += 1
			fragmentdict[k] = fragmentdict[k]*tf*idf(k)
			if fragmentdict[k] == 0:
				fragmentdict[k] = 0.0

	# d1dict and fragmentdict ready to be put into vectors
	global vocabList
	for v in vocabList:
		if d1:
			if v in d1dict:
				d1vector.append(d1dict[v])
			else:
				d1vector.append(0)

		if v in fragmentdict:
			fragmentvector.append(fragmentdict[v])
		else:
			fragmentvector.append(0)

	# Vectors ready
	cossim = cosinesim(d1vector,fragmentvector)
	return cossim

def usage():
	print "USAGE: python " + sys.argv[0] + " [-w] [-n <fragment size>] -1 <d1file> -2 <d2file>"
	print "-n: To specify size of fragments (by no. of lines). Default is 5"
	print "-w: To switch on with TF-IDF-weight mode. Default is False"
	print "<d1file> is the query file"
	print "<d2file> is the domain file"
	print "E.g. python " + sys.argv[0] + " -w -n 20 -1 search.txt -2 A00-1001.txt"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "wn:1:2:")
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
	except getopt.GetoptError:
		usage()
		sys.exit(2)

import cmd
class interactive(cmd.Cmd):
	"""Simple command processor example."""

	def do_df(self, term):
		"""USAGE:\tdf [term]
OUTPUT:\tRetrieve document frequency for [term]"""
		if term:
			global dfDict
			if term in dfDict:
				print dfDict[term]
			else:
				print "Record not found!"
		else:
			print "USAGE: df [term]"

	def do_idf(self, term):
		"""USAGE:\tidf [term]
OUTPUT:\tRetrieve inverse document frequency for [term]"""
		if term:
			global dfDict
			if term in dfDict:
				print (log(N) - log(int(dfDict[term])))
			else:
				print "Record not found!"
		else:
			print "USAGE: idf [term]"

	def do_tf(self, term):
		"""USAGE:\ttf [term]
OUTPUT:\tRetrieve term frequency for [term] in domain file"""	
		if term:
			if weightSwitch:
				global d2TFDict
				if term in d2TFDict:
					print len(d2TFDict[term])
					print "Lines: " + str(d2TFDict[term])
				else:
					print "Record not found!"
			else:
				print "TF file not loaded!"

	def do_score(self, term):
		"""USAGE:\tscore [|<fragment id>|max]
OUTPUT:\tPrints out the scores of the fragments"""
		global interactiveResults
		global resultsDict
		global top
		if term:
			if term.isdigit() and (int(term) in range(0, len(interactiveResults))):
				print "Fragment " + str(term) + ": " + str(interactiveResults[int(term)]) 
			elif term == "max":
				print "Fragment " + str(resultsDict[top[0]]) + " has the max score of " + str(top[0])
		else:
			for i in range(len(interactiveResults)):
				print "Fragment " + str(i) + ": " + str(interactiveResults[i])

	def do_print(self, line):
		"""USAGE:\tprint [-q |-f <fragment id> | -l <line no.>]
OUTPUT:\tPrints out the contents of query, fragment or line."""
		if line:
			tokens = line.split()
			if len(tokens)>2:
				print "USAGE:\tprint [-q |-f <fragment id> | -l <line no.>]"
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
								print d2FragmentsToTest[index]
							else:
								print "Fragment ID range: 0-" + str(len(d2FragmentsToTest))
						elif opt == "-l":
							global d2Lines
							index = int(args)
							if index>0 and index<len(d2Lines):
								print d2Lines[index-1]
							else:
								print "Line no. range: 1-" + str(len(d2Lines))
				except getopt.GetoptError:
					print "USAGE:\tprint [-q |-f <fragment id> | -l <line no.>]"
		else:
			print "USAGE:\tprint [-q |-f <fragment id> | -l <line no.>]"

	def do_quit(self, line):
		"""Exit the program"""
		print "Good Bye!"
		sys.exit()

	def do_EOF(self, line):
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
	print "-----\tEntering interactive mode\t-----"
	interactive().cmdloop()




