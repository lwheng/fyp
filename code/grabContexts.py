# To grab all contexts for each citation
# Eg. output: cite_key:<context a>
#             cite_key:<context b>

import re
import os
import sys
import string
import getopt
import myUtils
from xml.dom.minidom import parseString
import urllib2
from datetime import datetime

date = str(datetime.now().date())
time = str(datetime.now().time())

annotationMasterFile = ""

def grabContext():
  global annotationMasterFile
  openfile = open(annotationMasterFile, "r")
  for l in openfile:
    cite_key = myUtils.removespecialcharacters(l)[0:-1]
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]

    # cited paper
    citedFile = "http://wing.comp.nus.edu.sg/~antho/" + cited[0] + "/" + cited[0:3] + "/" + cited 
    titleCited = ""
    try:
      # try bib file first
      openfile = urllib2.urlopen(citedFile + ".bi", "r")
      regexTitle = r'(^title)\s*=\s*\{(.*)\}'
      for l in openfile:
        line = l.strip()
        matchObj = re.match(regexTitle, line, re.M|re.I)
        if matchObj:
          titleCited = matchObj.group(2)
      openfile.close()
    except urllib2.HTTPError, e:
      # bib file not file
      try:
        # now try -final.xml
        openfile = urllib2.urlopen(citedFile + "-final.xml", "r")
        data = openfile.read()
        openfile.close()
        dom = parseString(data)
        print dom.toxml()
      except urllib2.HTTPError, e:
        # seersuite file not found
        print "Error"
    
    # citing paper
    citingParscitFile = "http://wing.comp.nus.edu.sg/~antho/" + citing[0] + "/" + citing[0:3] + "/" + citing + "-parscit.xml"
    file = urllib2.urlopen(citingParscitFile, "r")
    data = file.read()
    file.close()
    dom = parseString(data)
    citationList = dom.getElementsByTagName('citationList')[0].getElementsByTagName('citation')
    for citation in citationList:
      title = citation.getElementsByTagName('title')[0]
      print title.toxml()
    sys.exit()

def usage():
  print "USAGE: python " + sys.argv[0] + " -f <annotationMasterFile>"
  print "Default output location is Downloads, grabContext-(timestamp).txt"

def main(argv):
  try:
    opts, args = getopt.getopt(argv, "f:")
    for opt, args in opts:
      if opt == "-f":
        global annotationMasterFile
        annotationMasterFile = args
  except getopt.GetoptError:
    usage()
    sys.exit(2)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    usage()
    sys.exit()
  main(sys.argv[1:])
  grabContext()
