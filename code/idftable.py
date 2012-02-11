import sys
import string	
from nltk.probability import FreqDist

idfname = sys.argv[1]
openidfname = open(idfname,"r")
idfdict = {}
for line in openidfname:
	l = line[:-1]
	idfdict[l]=0
# All words are now in dictionary
openidfname.close()

# Now open the file that contains the list of files
listname = sys.argv[2]
openlistname = open(listname, "r")

outputname = sys.argv[3]
openoutput = open(outputname,"w")

for f in openlistname:
	# filename is something like A00-1001.txt, need to open it again
	filename = f[:-1]
	print "Now we are at this file: " + filename
	# At this point, perform steps like papertowords to get bag
	openfile = open(filename,"r")
	bag = []
	for line in openfile:
		words = line.split()
		for w in words:
			# print "w is " + w
			word = w.lower()
			bag.append(word)
	openfile.close()
	voc = FreqDist(bag)
	# Now we have the voc, ready to update the idf dictionary
	for w in voc:
		if w in idfdict:
			number = idfdict[w]
			number = number + 1
			idfdict[w] = number
openlistname.close()
	
for key in idfdict:
	towrite = key + "=====>" + str(idfdict[key]) + "\n"
	openoutput.write(towrite)
openoutput.close()
	
