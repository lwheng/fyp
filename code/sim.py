# We need the dictionary, for the set of all words in the corpus
import sys
import math 

dictFilename = ""
d1Filename = ""
d2Filename = ""
dictList = []
d1Vector = []
d2Vector = []

def usage():
	print "USAGE: python " + sys.argv[0] +" -d <dictfile> -1 <d1file> -2 <d2file>"

import getopt
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:1:2:")
        for opt, args in opts:
            if opt == "-d":
                global dictFilename
                dictFilename = args
            elif opt == "-1":
                global d1Filename
                d1Filename = args
            elif opt == "-2":
                global d2Filename
                d2Filename = args
    except getopt.GetoptError:
    	usage()
    	sys.exit(2)


def addFilesToMemory():
	dictionary = dictFilename
	opendictionary = open(dictionary,"r")
	# Now read dictionary into memory
	global dictList
	dictList = []
	for l in opendictionary:
		line = l[:-1]
		dictList.append(line)

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

	# Read d1 and d2 into memory
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
	if (len(sys.argv) != 7):
		usage()
	else:
		main(sys.argv[1:])
		addFilesToMemory()
		sim(d1Filename,d2Filename)












