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
import Levenshtein
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
      # bib file not file, now try parscit-section.xml
      try:
        openfile = urllib2.urlopen(citedFile + "-parscit.xml", "r")
        data = openfile.read()
        openfile.close()
        dom = parseString(data)
        title = dom.getElementsByTagName('title')[0]
        titleCited = title.firstChild.nodeValue
      except urllib2.HTTPError, e:
        try:
          # now try -final.xml
          openfile = urllib2.urlopen(citedFile + "-final.xml", "r")
          data = openfile.read()
          openfile.close()
          dom = parseString(data)
          title = dom.getElementsByTagName('teiHeader')[0].getElementsByTagName('fileDesc')[0].getElementsByTagName('titleStmt')[0].getElementsByTagName('title')[0]
          titleCited = title.firstChild.nodeValue
        except urllib2.HTTPError, e:
          # file not found
          print "Error: For " + cite_key + ", cannot get title for cited " + cited
    
    # citing paper
    citingParscitFile = "http://wing.comp.nus.edu.sg/~antho/" + citing[0] + "/" + citing[0:3] + "/" + citing + "-parscit.xml"
    try:
      file = urllib2.urlopen(citingParscitFile, "r")
      data = file.read()
      file.close()
      dom = parseString(data)

      if dom.getElementsByTagName('citationList')[0].getElementsByTagName('citation'):
        citationList = dom.getElementsByTagName('citationList')[0].getElementsByTagName('citation')

        bestIndex = 0
        maxRatio = 0
        for i in range(len(citationList)):
          citation = citationList[i]
          if citation.attributes['valid'].value == 'true':
            if citation.getElementsByTagName('title'):
              title = citation.getElementsByTagName('title')[0].firstChild.nodeValue
            elif citation.getElementsByTagName('booktitle'):
              title = citation.getElementsByTagName('booktitle')[0].firstChild.nodeValue
            elif citation.getElementsByTagName('journal'):
              title = citation.getElementsByTagName('journal')[0].firstChild.nodeValue
            elif citation.getElementsByTagName('note'):
              title = citation.getElementsByTagName('note')[0].firstChild.nodeValue
            ratio = Levenshtein.ratio(title, titleCited)
            if ratio > maxRatio:
              bestIndex = i
              maxRatio = ratio
        citation = citationList[bestIndex]
        if citation.getElementsByTagName('contexts'):
          contexts = citation.getElementsByTagName('contexts')[0].getElementsByTagName('context')
          for context in contexts:
            print cite_key + "!=" + context.toxml()
        else:
          print cite_key + "!=" + ""
      else:
        print cite_key + "!=" + ""
    except urllib2.HTTPError, e:
      print "Error"
    # sys.exit() # to terminate loop after first line

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
