# INPUT: ACL ARC Interlink File
# OUTPUT: Cited==>Citees...

import sys

interlinkFile = ""
interlinkHash = {}

outputFile = "/Users/lwheng/Dropbox/fyp/annotation/cited-citees.txt"

def cited_citees():
  global interlinkFile
  global interlinkHash
  openfile = open(interlinkFile, "r")
  for l in openfile:
    line = l.replace("\n","").replace(" ","")
    tokens = line.split("==>")
    key = tokens[1]
    value = tokens[0]
    if key not in interlinkHash:
      vList = []
      vList.append(value)
      interlinkHash[key] = vList
    else:
      interlinkHash[key].append(value)

  openfile.close()

  writefile = open(outputFile, "w")
  for k in interlinkHash:
    towrite = k + "==>" + str(interlinkHash[k]) + "\n"
    writefile.write(towrite)
  writefile.close()

def usage():
  print "Usage"

def main(argv):
  global interlinkFile
  interlinkFile = argv[-1]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    usage()
    sys.exit(2)
  main(sys.argv[1:])
  cited_citees()
