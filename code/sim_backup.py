import sys
import math 

vocabFilename = ""
d1Filename = ""
d2Filename = ""
dictList = []
d1Vector = []
d2Vector = []
weightSwitch = False
N = 8889

# If weightSwitch is True
# The filenames of the DF and TF tables
# Amend accordingly
dftablfilename = "/Users/lwheng/Desktop/dftable.txt"
tftablefilename = "/Users/lwheng/Desktop/tftablestripped.txt"
dfDict = {}
tfDict = {}

def usage():
	print "USAGE: python " + sys.argv[0] +" -w -v <vocabfile> -1 <d1file> -2 <d2file>"
	print "-w: To switch on with TF-IDF-weight mode"
	print "<vocabfile> contains the set of all words in the corpus"
	print "<d1file> is the query file"
	print "<d2file> is the domain file"

import getopt
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "wv:1:2:")
        for opt, args in opts:
            if opt == "-v":
                global vocabFilename
                vocabFilename = args
            elif opt == "-1":
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


def loadDictionaryToMemory():
	dictionary = vocabFilename
	opendictionary = open(dictionary,"r")
	# Now read dictionary into memory
	global dictList
	dictList = []
	for l in opendictionary:
		line = l[:-1]
		dictList.append(line)
	opendictionary.close()

def loadDFTableToMemory():
	opendftable = open(dftablfilename,"r")
	global dfDict
	for l in opendftable:
		line = l[:-1]
		tokens = line.split("\t")
		dfDict[tokens[0]] = tokens[1]
	opendftable.close()

def loadTFTableToMemory():
	opentftable = open(tftablefilename,"r")
	global tfDict
	previousKey = ""
	for l in opentftable:
		line = l[:-1]
		tokens = line.split("\t")
		if len(tokens) == 3:
			if tokens[0] != previousKey:
				tfDict[tokens[0]] = {}
				tfDict[tokens[0]][tokens[1]] = tokens[2]
				previousKey = tokens[0]
			else:
				tfDict[tokens[0]][tokens[1]] = tokens[2]
	opentftable.close()

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
	# len(v1) must be equal to len(v2) (You don't say!)
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

def tf(term, doc):
	global tfDict
	if doc in tfDict:
		if term in tfDict[doc]:
			return tfDict[doc][term]
		else:
			print "term not in tfDict[doc]"
	else:
		print "doc not in tfDict"
		print doc
	# if tfDict[doc]:
	# 	if tfDict[doc][term]:
	# 		return tfDict[doc][term]
	return 0

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

	# Original way
	# return dotproduct(v1,v2)/(magnitude(v1)*magnitude(v2))

def sim(d1, d2):
	opend1 = open(d1, "r")
	opend2 = open(d2, "r")

	global d1Vector
	global d2Vector
	d1Vector = []
	d2Vector = []
	d1Dict = {}
	d2Dict = {}

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
		fileExtension = ".txt"
		for term in dictList:
			if term in d1Dict:
				idfVal = idf(N, int(df(term)))
				dTokens = d2Filename.split("/")
				d = dTokens[-1][:-len(fileExtension)]
				tfVal = int(tf(term, d))
				tfidf = tfVal * idfVal
				d1Vector.append(tfidf)
			else:
				d1Vector.append(0)

			if term in d2Dict:
				idfVal = idf(N, int(df(term)))
				dTokens = d2Filename.split("/")
				d = dTokens[-1][:-len(fileExtension)]
				tfVal = int(tf(term, d))
				tfidf = tfVal * idfVal
				d2Vector.append(tfidf)
			else:
				d2Vector.append(0)
	else:
		for term in dictList:
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
	loadDictionaryToMemory()
	if weightSwitch:
		loadDFTableToMemory()
		loadTFTableToMemory()
	sim(d1Filename,d2Filename)
	sys.exit()












