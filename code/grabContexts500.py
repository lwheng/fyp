import re
import os
import sys
import string
import getopt
import myUtils
from xml.dom.minidom import parseString
import urllib2
import Levenshtein
import unicodedata

annotations500File = ""

def grabContext():
  global annotations500File
  openfile = open(annotations500File, "r")
  for l in openfile:
    cite_key = myUtils.removespecialcharacters(l)
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]

    # cited paper
    citedFile = "/Users/lwheng/Downloads/fyp/annotations500/" + cited
    titleCited = ""
    if os.path.exists(citedFile + ".bib"):
      openthefile = open(citedFile + ".bib")
      regexTitle = r"(^title)\s*=\s*\{(.*)\}"
      for l in openthefile:
        line = l.strip()
        matchObj = re.match(regexTitle, line, re.M|re.I)
        if matchObj:
          titleCited = matchObj.group(2)
      openthefile.close()
    elif os.path.exists(citedFile + "-parscit-section.xml"):
      openthefile = open(citedFile + "-parscit-section.xml")
      data = openthefile.read()
      openthefile.close()
      dom = parseString(data)
      node = dom.getElementsByTagName('title')
      if node:
        titleCited = node[0].firstChild.nodeValue.strip()
      else:
        node = dom.getElementsByTagName('note')
        if node:
          titleCited = node[0].firstChild.nodeValue.strip()
    elif os.path.exists(citedFile + "-final.xml"):
      openthefile = open(citedFile + "-final.xml")
      data = openthefile.read()
      openthefile.close()
      dom = parseString(data)
      node = dom.getElementsByTagName('teiHeader')[0].getElementsByTagName('fileDesc')[0].getElementsByTagName('titleStmt')[0].getElementsByTagName('title')[0]
      titleCited = node.firstChild.nodeValue
    else:
      print "title not found"
    titleCited = titleCited.replace("\n", " ")
    titleCited = titleCited.capitalize()

    # citing paper
    citingParscitFile = "/Users/lwheng/Downloads/fyp/annotations500/" + citing + "-parscit.xml"
    openthefile = open(citingParscitFile, "r")
    data = openthefile.read()
    openthefile.close()
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
          title = unicodedata.normalize("NFKD", unicode(title)).encode('ascii', 'ignore')
          titleCited = unicodedata.normalize("NFKD", unicode(titleCited)).encode('ascii', 'ignore')
          ratio = Levenshtein.ratio(str(title), str(titleCited))
          if ratio > maxRatio:
            bestIndex = i
            maxRatio = ratio
      citation = citationList[bestIndex]
      if citation.getElementsByTagName('contexts'):
        contexts = citation.getElementsByTagName('contexts')[0].getElementsByTagName('context')
        for context in contexts:
          print cite_key + "!=" + unicodedata.normalize("NFKD", context.toxml()).encode('ascii','ignore')
      else:
        print cite_key + "!=" + "NO CONTEXT"
    else:
      print cite_key + "!=" + "NO CITATION"

def usage():
  print "USAGE: python " + sys.argv[0] + " -f <annotations500File>"

def main(argv):
  try:
    opts, args = getopt.getopt(argv, "f:")
    for opt, args in opts:
      if opt == "-f":
        global annotations500File
        annotations500File = args
  except getopt.GetoptError:
    usage()
    sys.exit(2)

if __name__ == '__main__':
  if len(sys.argv) < 3:
    usage()
    sys.exit(2)
  main(sys.argv[1:])
  grabContext()
