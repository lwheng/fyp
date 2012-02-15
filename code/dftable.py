import sys
import string	
from nltk.probability import FreqDist

dfname = sys.argv[1]
opendfname = open(dfname,"r")
dfdict = {}
for line in opendfname:
	l = line[:-1]
	dfdict[l]=0
# All words are now in dictionary
opendfname.close()

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
			# print "word is: " + word
			bag.append(word)
	openfile.close()
	voc = FreqDist(bag)
	# Now we have the voc, ready to update the idf dictionary
	for w in voc:
		if w in dfdict:
			number = dfdict[w]
			number = number + 1
			# print "(w,number): (" + w + ", " + str(number) + ")"
			dfdict[w] = number
openlistname.close()
	
for key in dfdict:
	towrite = key + "=====>" + str(dfdict[key]) + "\n"
	print "towrite :" + towrite
	openoutput.write(towrite)
openoutput.close()
	
