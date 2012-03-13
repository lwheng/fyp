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
import myUtils

fileDirectory = ""
tfDirectory = "/Users/lwheng/Download/tf"
if not os.path.isdir(tfDirectory):
	os.makedirs(tfDirectory)

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