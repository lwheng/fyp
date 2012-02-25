# Computes cosine similarity,
# with the option to use tf-idf weighting

# Need:
# vocabfile
# dftable
# tfDirectory

# Input:
# Query file
# Domain file

# Output:
# Value of sim(d1, d2)

import sys
import os
import math
import string
import getopt

# No. of documents in dataset
N = 500

mainDirectory = "/Users/lwheng/Desktop"
dfFile = "dftable-(2012-02-25-15:53:31.360114).txt"
vocabFile = "vocab-(2012-02-24-23:03:33.252021).txt"

tfDirectory = str(os.path.join(mainDirectory, "tf"))
dfPath = str(os.path.join(mainDirectory, dfFile))
vocabPath = str(os.path.join(mainDirectory,vocabFile))

vocabList = []
dfDict = {}
tfDict = {}

weightSwitch = False

d1Filename = ""
d2Filename = ""

def loadVocab():
	global vocabList
	global vocabPath
	openvocab = open(vocabPath,"r")
	for l in openvocab:
		line = l[:-1]
		vocabList.append(line)
	openvocab.close()

def loadDFTable():
	global dfDict
	opendf = open(dfPath,"r")
	for l in opendf:
		line = l[:-1]
		tokens = line.split()
		dfDict[tokens[0]] = tokens[1]
	opendf.close()

def loadTFTable():
	global d2Filename
	global tfDict
	tokens = d2Filename.split("/")
	tfFilename = tokens[-1].split(".txt")[0] + ".tf"
	count = 1
	for dirname, dirnames, filenames in os.walk(tfDirectory):
		if tfFilename in filenames:
			tfPath = str(os.path.join(tfDirectory, tfFilename))
			opentf = open(tfPath,"r")
			for l in opentf:
				line = l[:-1]
				tokens = line.split()
				term = tokens[0]
				locations = tokens[1].split("!")
				tfDict[term] = locations
			opentf.close()
		else:
			print "TF for " + tokens[-1] + " not found."
			print "Reverting back to weigtless mode"
			global weightSwitch
			weightSwitch = False

def magnitude(v):
	# Computes Euclidean Length of vector
	output = 0
	sumofsquares = 0
	for i in v:
		sumofsquares = sumofsquares + i**2
	output = math.sqrt(sumofsquares)
	return output

def dotproduct(v1, v2):
	# Computes dot product of 2 vectors
	if (len(v1) != len(v2)):
		return False
	output = 0
	for i in range(len(v1)):
		output = output + (v1[i]*v2[i])
	return output

def cosine(x):
	# Return the cosine of x radians
	return math.cos(x)

def log(x):
	# Return the natual log of x
	return math.log(x)

def df(term):
	global dfDict
	if (dfDict[term]):
		return dfDict[term]
	else:
		return 0

def idf(N, df):
	return log(N/df)

def cosinesim(v1, v2):
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
		sumofsquares1 = sumofsquares1 + v1[i]**2
		sumofsquares2 = sumofsquares2 + v2[i]**2
	
	return dot/((math.sqrt(sumofsquares1))*(math.sqrt(sumofsquares2)))

def usage():
	print "USAGE: python " + sys.argv[0] +" -w -1 <d1file> -2 <d2file>"
	print "-w: To switch on with TF-IDF-weight mode"
	print "<d1file> is the query file"
	print "<d2file> is the domain file"

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "w1:2:")
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
    except getopt.GetoptError:
    	usage()
    	sys.exit(2)

def sim(d1,d2):
	opend1 = open(d1, "r")
	opend2 = open(d2, "r")
	d1Dict = {}
	d2Dict = {}
	d1Vector = []
	d2Vector = []

	global N

	# Load d1 and d2 into memory
	for l in opend1:
		if l[-1] == "\n":
			line = l[:-1]
		else:
			line = l
		line = line.lower()
		tokens = line.split()
		for t in tokens:
			if t not in d1Dict:
				d1Dict[t] = 0
			d1Dict[t] += 1
	opend1.close()

	for l in opend2:
		if l[-1] == "\n":
			line = l[:-1]
		else:
			line = l
		line = line.lower()
		tokens = line.split()
		for t in tokens:
			if t not in d2Dict:
				d2Dict[t] = 0
			d2Dict[t] += 1
	opend2.close()

	if weightSwitch:
		# Use dftable and tftable
		for term in vocabList:
			if term in d1Dict:
				idfval = idf(N, float(df(term)))
				if term in tfDict:
					tfidf = len(tfDict[term]) * idfval
				else:
					tfidf = 0

				# What if idfval is negative? ---> FIX THIS
				if tfidf < 0:
					tfidf = 0
				d1Vector.append(tfidf)
			else:
				d1Vector.append(0)

			if term in d2Dict:
				idfval = idf(N, float(df(term)))
				if term in tfDict:
					tfidf = len(tfDict[term]) * idfval
				else:
					tfidf = 0

				# What if idfval is negative? ---> FIX THIS
				if tfidf < 0:
					tfidf = 0
				d2Vector.append(tfidf)
			else:
				d2Vector.append(0)

	else:
		for term in vocabList:
			if term in d1Dict:
				d1Vector.append(d1Dict[term])
			else:
				d1Vector.append(0)

			if term in d2Dict:
				d2Vector.append(d2Dict[term])
			else:
				d2Vector.append(0)
	cossim = cosinesim(d1Vector,d2Vector)
	print cossim

if __name__ == '__main__':
	main(sys.argv[1:])
	loadVocab()
	if weightSwitch:
		loadDFTable()
		loadTFTable()
	sim(d1Filename,d2Filename)
	sys.exit()