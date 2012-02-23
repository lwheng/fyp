# This script takes a set of documents and generates the
# Term Frequency tables

import sys

# File that contains the list of files
listname = sys.argv[1]
openlistname = open(listname,"r")

# This is the master dict
tf = {}

for f in openlistname:
	filename = f[:-1]
	filestokens = filename.split("/")
	file_id = filestokens[-1]
	new_dict = {}
	openfilename = open(filename,"r")
	print "Now at file: " + filename
	for l in openfilename:
		line = l[:-1].lower()
		words = line.split()
		for w in words:
			# new_bag.append(w)
			if (w not in new_dict):
				new_dict[w] = 0
			new_dict[w] += 1
	openfilename.close()
	tf[file_id] = new_dict
openlistname.close()

outputname = sys.argv[2]
openoutput = open(outputname,"w")

for key in tf:
	tempdict = tf[key]
	for k in tempdict:
		towrite = key + "\t" + k + "\t" + str(tempdict[k]) + "\n"
		print towrite
		openoutput.write(towrite)
# import pickle
# pickle.dump(tf, openoutput)
openoutput.close()


