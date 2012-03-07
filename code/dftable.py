# This script takes in a set of documents and generates
# the Document Frequency table.

# Needs
# 1. vocab
# 2. Document directory

import os
import sys
import string   
import getopt
import myUtils

from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

fileDirectory = ""
dftableFile = "/Users/lwheng/Desktop/dftable-(" + date + "-" + time + ").txt"

dftableDict = {}

def loadVocabFile():
	global vocabFile
	global dftableDict
	openvocab = open(vocabFile,"r")
	for l in openvocab:
		line = l[:-1]
		dftableDict[line] = 0
	openvocab.close()

def dftable():
	global fileDirectory
	global dftableFile

	for dirname, dirnames, filenames in os.walk(fileDirectory):
		for filename in filenames:
			thefile = str(os.path.join(dirname, filename))
			openthefile = open(thefile,"r")
			fileDict = {}
			for l in openthefile:
				line = myUtils.removespecialcharacters(l)
				line = line.lower()
				tokens = line.split()
				for t in tokens:
					toadd = ""
					if t.isalnum() or myUtils.hyphenated(t) or myUtils.apos(t):
						toadd = t
					elif (myUtils.removepunctuation(t)).isalnum():
						toadd = myUtils.removepunctuation(t)

					if len(toadd) != 0:
						if toadd not in fileDict:
							fileDict[toadd] = 0
						fileDict[toadd] += 1
			for key in fileDict:
				if key in dftableDict:
					dftableDict[key] += 1
			openthefile.close()
	openoutput = open(dftableFile,"w")
	for k in dftableDict:
		towrite = k + "\t" + str(dftableDict[k]) + "\n"
		openoutput.write(towrite)
	openoutput.close()

def usage():
	print "USAGE: python " + sys.argv[0] +" -v <vocab file> -d <fileDirectory>"
	print "Default output location is Desktop, dftable-(timestamp).txt"
	print "To specify output, add this: -o <output filename>"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "v:d:o:")
		for opt, args in opts:
			if opt == "-d":
				global fileDirectory
				fileDirectory = args
			elif opt == "-v":
				global vocabFile
				vocabFile = args
			elif opt == "-o":
				global dftableFile
				dftableFile = args
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	main(sys.argv[1:])
	loadVocabFile()
	dftable()
	
