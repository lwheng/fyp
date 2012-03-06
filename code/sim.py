import sys
import os
import math
import string
import getopt
import myUtils

# No. of documents in corpora
N = 500

mainDirectory = "/Users/lwheng/Desktop"
dfFile = "dftable-(2012-03-06-14:43:35.454643).txt"
vocabFile = "vocab-(2012-03-06-14:43:19.751722).txt"

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
	loadDFTable()
	if weightSwitch:
		loadTFTable()
	print "-----\tLoading Done!\t-----"

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
	d2Fragments = []
	for i in xrange(0, len(d2), fragmentSize):
		d2Fragments.append(d2[i:i+fragmentSize])

	# Overlap the fragments
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


	print "-----\tMax Sim() Computed!\t-----"
	print "-----\tResults\t-----"

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

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	loadFiles()
	maxsim(d1Lines,d2Lines)
	sys.exit()