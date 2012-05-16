import sys
import string

filename = sys.argv[1]
newfilename = sys.argv[2]

openfile = open(filename,"r")
opennewfile = open(newfilename,"w")

for l in openfile:
	line = l[:-1]
	line = line.lower()
	line = line + "\n"
	opennewfile.write(line)
openfile.close()
opennewfile.close()
	
