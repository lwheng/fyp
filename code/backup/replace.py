import sys

filename = sys.argv[1]
openfilename = open(filename, "r")

outputname = sys.argv[2]
openoutput = open(outputname,"w")

for l in openfilename:
	line = l
	if l[-1] == "\n":
		line = l[:-1]
	tokens = line.split(".txt")
	towrite = tokens[0] + tokens[1] + "\n"
	openoutput.write(towrite)