import cPickle as pickle

titleHash = {}
titleFile = "/Users/lwheng/Dropbox/fyp/annotation/paperTitles.txt"
openfile = open(titleFile, "r")
for l in openfile:
	line = l.strip()
	info = line.split("==>")
	titleHash[info[0]] = info[1]
openfile.close()

pickle.dump(titleHash, open("paperTitles.pickle", "wb"))