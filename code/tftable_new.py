# This script takes in a set of documents and generates
# the Term Frequency Table for each file

# Input
# Set of documents
# Output
# Set of .tf files

import os
import sys
import string   
import getopt

fileDirectory = ""
tfDirectory = "/Users/lwheng/Desktop/tf"

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

def tftable():
	global fileDirectory
	global tfDirectory

	for dirname, dirnames, filenames in os.walk(fileDirectory):
		for filename in filenames:
			tffilename = filename.replace(".txt", ".tf")
			inputName = str(os.path.join(fileDirectory, filename))
			outputName = str(os.path.join(tfDirectory, tffilename))
			
			openinput = open(inputName,"r")
			fileDict = {}
			linenumber = 1
			for l in openinput:
				line = l[:-1]
				line = line.lower()
				tokens = line.split()
				for t in tokens:
					toadd = ""
					if t.isalnum() or hyphenated(t) or apos(t):
						toadd = t
					elif removepunctuation(t).isalnum():
						toadd = removepunctuation(t)
					if len(toadd) > 0:
						if not toadd in fileDict:
							fileDict[toadd] = []
						fileDict[toadd].append(linenumber)
				linenumber += 1
			openinput.close()
			openoutput = open(outputName,"w")
			for k in fileDict:
				locations = fileDict[k]
				towrite = k + "\t"
				for l in locations:
					towrite += str(l) + "!"
				towrite = towrite[:-1] + "\n"
				openoutput.write(towrite)
			openoutput.close()

def usage():
	print "USAGE: python " + sys.argv[0] +" -d <fileDirectory>"

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "d:")
		for opt, args in opts:
			if opt == "-d":
				global fileDirectory
				fileDirectory = args
	except getopt.GetoptError:
		usage()
		sys.exit(2)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		usage()
		sys.exit()
	main(sys.argv[1:])
	tftable()