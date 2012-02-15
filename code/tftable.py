import sys
from nltk.probability import FreqDist

# File that contains the list of files
listname = sys.argv[1]
openlistname = open(listname,"r")

# File that contains the list of terms
terms = sys.argv[2]
openterms = open(terms,"r")
termlist = []
for t in openterms:
	termlist.append(t[:-1])

# This is the master dict
tf = {}

for f in openlistname:
	filename = f[:-1]
	filestokens = filename.split("/")
	file_id = filestokens[-1]
	new_dict = {}
	new_bag = []
	openfilename = open(filename,"r")
	# print "Now at file: " + filename
	for l in openfilename:
		line = l[:-1].lower()
		words = line.split()
		for w in words:
			new_bag.append(w)
	openfilename.close()
	voc = FreqDist(new_bag)
	for w in termlist:
		# print w+ ":  " + str(voc[w])
		if w in voc:
			new_dict[w] = voc[w]
		else:
			new_dict[w] = 0
	# print new_dict
	tf[file_id] = new_dict
openlistname.close()


