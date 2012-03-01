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

# No. of documents in corpora
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
fragmentSize = 5

d1Filename = ""
d2Filename = ""

d1Lines = []
d2Lines = []

d2DFTable = {}

def loadD1():
	global d1Filename
	global d1Lines
	opend1 = open(d1Filename,"r")
	for l in opend1:
		line = l
		line = line.lower()
		line = line.replace("\n","")
		line = line.replace("\t","")
		line = line.replace("\r","")
		if len(line) != 0:
			d1Lines.append(line)
	opend1.close()

def loadD2():
	# Now D2 is the "corpus", each fragment (e.g. paragraph)
	# is a "document" in this "corpus"
	# So, we need DFTable and TFTable for this new "corpus"
	global d2Filename
	global d2Lines
	opend2 = open(d2Filename,"r")
	for l in opend2:
		line = l
		line = line.lower()
		line = line.replace("\n","")
		line = line.replace("\t","")
		line = line.replace("\r","")
		if len(line) != 0:
			d2Lines.append(line)
	opend2.close()

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
			print "Reverting back to weightless mode"
			global weightSwitch
			weightSwitch = False

def hyphenated(word):
    tokens = word.split("-")
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def apos(word):
    tokens = word.split("'")
    if len(tokens) != 2:
        return False
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def removepunctuation(word):
    output = ""
    for w in word:
        if not w in string.punctuation:
            output += w
    return output

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
		sumofsquares1 = sumofsquares1 + (v1[i])**2
		sumofsquares2 = sumofsquares2 + (v2[i])**2

	if sumofsquares1==0 or sumofsquares2==0:
		return 0
	
	return dot/((math.sqrt(sumofsquares1))*(math.sqrt(sumofsquares2)))

def usage():
	print "USAGE: python " + sys.argv[0] +" [-w] [-n <fragment size>] -1 <d1file> -2 <d2file>"
	print "-n: To specify size of fragments (by no. of lines). Default is 5"
	print "-w: To switch on with TF-IDF-weight mode. Default is False"
	print "<d1file> is the query file"
	print "<d2file> is the domain file"

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

def maxsim(d1,d2):
	# d1 is query file in lines
	# d2 is the domain file in lines,
	# now the "corpora". To be divided into "documents" (fragments)
	
	# Now we divide d2 into fragments
	global fragmentSize
	d2Fragments = []
	for i in xrange(0, len(d2), fragmentSize):
		d2Fragments.append(d2[i:i+fragmentSize])

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

	# We need to compute the DFTable just for this "corpus"
	global d2DFTable
	d2Lines = {}
	for i in range(len(d2)):
		l = d2[i]
		line = l.lower()
		line = l.replace("\n", "")
		line = l.replace("\r", "")
		line = l.replace("\t", "")
		if len(line) != 0:
			d2Lines[line] = i+1

		tokens = line.split()
		for t in tokens:
			toadd = ""
			if t.isalnum() or hyphenated(t) or apos(t):
				toadd = t
			elif removepunctuation(t).isalnum():
				toadd = removepunctuation(t)

			if len(toadd) != 0:
				if toadd not in d2DFTable:
					d2DFTable[toadd] = 0
				d2DFTable[toadd] += 1

	# Now we find the maxsim
	resultsList = []
	fragmentsCount = len(d2FragmentsToTest)
	# for fragment in d2FragmentsToTest:
		# resultsList.append(sim(d1, fragment, len(d2FragmentsToTest)))
	for i in range(len(d2FragmentsToTest)):
		resultsList.append(sim(d1, d2FragmentsToTest[i], fragmentsCount, range(i*fragmentSize+1, (i+1)*fragmentSize+1)))

	maxScore = 0
	fragmentMax = 0
	print "Total no. of fragments:\t" + str(fragmentsCount)
	print "Fragment Scores:"
	for i in range(len(resultsList)):
		if resultsList[i] != 0:
			print "Fragment " + str(i) + "\t" + str(resultsList[i])
			if resultsList[i] > maxScore:
				maxScore = resultsList[i]
				fragmentMax = i

	if maxScore == 0:
		print "No fragments match!!"
		print "------------------------------"
		print "The search query did not match any of the fragments. Score is 0.0"
	else:
		print "------------------------------"
		print "Fragment " + str(fragmentMax) + " has the highest score of " + str(maxScore)
		print "------------------------------"
		print "Contents of fragment " + str(fragmentMax) + ":"
		print d2FragmentsToTest[fragmentMax]
		print "------------------------------"
		print "Location of fragment in domain document:"
		print "This fragment is found from line " + str(d2Lines[d2FragmentsToTest[fragmentMax][0]]) \
			+ "-" + str(d2Lines[d2FragmentsToTest[fragmentMax][-1]]) \
			+ " of the domain document"


def sim(d1,fragment,fragmentsCount,lineRange):
	# d1 is query in lines
	# fragment is a fragment in lines
	# fragmentsCount is no. of fragments

	d1Dict = {}
	fragmentDict = {}
	d1Vector = []
	fragmentVector = []

	# We use the actual corpus's vocab
	# Why? Because it simply has more words
	global vocabList
	global d2DFTable

	for l in d1:
		tokens = l.split()
		for t in tokens:
			toadd = ""
			if t.isalnum() or hyphenated(t) or apos(t):
				toadd = t
			elif removepunctuation(t).isalnum():
				toadd = removepunctuation(t)

			if len(toadd) != 0:
				if toadd not in d1Dict:
					d1Dict[toadd] = 0
				d1Dict[toadd] += 1
	for l in fragment:
		tokens = l.split()
		for t in tokens:
			toadd = ""
			if t.isalnum() or hyphenated(t) or apos(t):
				toadd = t
			elif removepunctuation(t).isalnum():
				toadd = removepunctuation(t)

			if len(toadd) != 0:
				if toadd not in fragmentDict:
					fragmentDict[toadd] = 0
				fragmentDict[toadd] += 1

	if weightSwitch:
		# DFTable just for this "corpus" is computed by maxsim
		# TFTable for this "corpus" we already have
		for term in vocabList:
			if (term in d1Dict) and (term in d2DFTable):
				idfval = idf(fragmentsCount, float(d2DFTable[term]))
				if term in tfDict:
					locations = tfDict[term]
					tf = 0
					for location in locations:
						if int(location) in lineRange:
							tf += 1
					tfidf = tf * idfval
				else:
					tfidf = 0

				if tfidf < 0:
					tfidf = 0
				d1Vector.append(tfidf)
			else:
				d1Vector.append(0)

			if (term in fragmentDict) and (term in d2DFTable):
				idfval = idf(fragmentsCount, float(d2DFTable[term]))
				if term in tfDict:
					locations = tfDict[term]
					tf = 0
					for location in locations:
						if int(location) in lineRange:
							tf += 1
					tfidf = tf * idfval
				else:
					tfidf = 0

				if tfidf < 0:
					tfidf = 0
				fragmentVector.append(tfidf)
			else:
				fragmentVector.append(0)
	else:
		for term in vocabList:
			if term in d1Dict:
				d1Vector.append(d1Dict[term])
			else:
				d1Vector.append(0)

			if term in fragmentDict:
				fragmentVector.append(fragmentDict[term])
			else:
				fragmentVector.append(0)

	cossim = cosinesim(d1Vector,fragmentVector)
	return cossim

if __name__ == '__main__':
	if len(sys.argv) < 3:
		usage()
		sys.exit()
	main(sys.argv[1:])
	loadVocab()
	loadD1()
	loadD2()
	if weightSwitch:
		# loadDFTable()
		loadTFTable()
	maxsim(d1Lines,d2Lines)
	sys.exit()