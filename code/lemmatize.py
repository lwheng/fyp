# INPUT: A set of documents
# OUTPUT: The set of documents, but with words lemmatized

import os
import sys
import string
import getopt
import myUtils
from nltk.stem.wordnet import WordNetLemmatizer

lm = WordNetLemmatizer()
inDirectory = ""
outDirectory = ""

def lemma():
	global inDirectory
	global outDirectory
	print inDirectory
	print outDirectory

	if not os.path.isdir(outDirectory):
		os.makedirs(outDirectory)

	for dirname, dirnames, filenames in os.walk(inDirectory):
		for f in filenames:
			filePath = str(os.path.join(inDirectory,f))
			print filePath
			outPath = str(os.path.join(outDirectory,f))

			readfile = open(filePath,"r")
			writefile = open(outPath,"w")
			for l in readfile:
				tokens = l.split()
				ready = map(lambda x:lm.lemmatize(myUtils.removespecialcharacters(x.lower())),tokens)
				towrite=""
				for r in ready:
					towrite += r + " "
				towrite = towrite[:-1] + "\n"
				writefile.write(towrite)
			readfile.close()
			writefile.close()



	# openfile = open(filename,"r")
	# # writefile = open(outputname,w)

	# for l in openfile:
	# 	towrite = ""
	# 	tokens = l.split()
	# 	for t in tokens:
	# 		totest = t.lower()
	# 		towrite += lm.lemmatize(totest) + " "
	# 	print towrite
	# openfile.close()

def usage():
	print "USAGE: python " + sys.argv[0] +" -d <fileDirectory> -o <outDirectory"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "d:o:")
		for opt, args in opts:
			if opt == "-d":
				global inDirectory
				inDirectory = args
			elif opt == "-o":
				global outDirectory
				outDirectory = args
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	lemma()
