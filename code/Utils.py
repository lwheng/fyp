from xml.dom.minidom import parseString
from xml.dom import Node
import unicodedata
import sys
import nltk
import re
import os
import math
import numpy
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance

class nltk_tools:
  def nltkWordTokenize(self, text):
    return nltk.word_tokenize(text)

  def nltkText(self, tokens):
    return nltk.Text(tokens)

  def nltkTextCollection(self, documents):
    return nltk.TextCollection(documents)

  def nltkStopwords(self):
    return stopwords.words('english')

  def nltkCosineDistance(self, u, v):
    return nltk.cluster.util.cosine_distance(u,v)

class tools:
  def parseXML(self, data):
    return parseString(data)

  def normalize(self, text):
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore')

  def searchTermInLines(self, term, lines):
    for i in range(len(lines)):
      l = lines[i]
      if term in l:
        return i
    return None

class weight:
  def __init__(self):
    self.sentenceTokenizer = PunktSentenceTokenizer()

  def chunkAverageWeight(self, chunk, collection):
    tempWeight = 0
    if len(chunk.tokens) == 0:
      return 0
    for t in chunk.tokens:
      tempWeight += collection.tf_idf(t.lower(), chunk)
    return float(tempWeight) / float(len(chunk.tokens))

  def citDensity(self, context_lines, context_citStr):
    # Regular Expression
    reg = []
    reg.append(r"\(\s?(\d{1,3})\s?\)")
    reg.append(r"\(\s?(\d{4})\s?\)")
    reg.append(r"\(\s?(\d{4};?\s?)+\s?")
    reg.append(r"\[\s?(\d{1,3}\s?,?\s?)+\s?\]")
    reg.append(r"\[\s?([\w-],?\s?)+\s?\]")
    reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?(et al\.?)?\s?,?\s?(\(?(\d{4})\)?)")
    regex = ""
    for i in range(len(reg)):
      regex += reg[i] + "|"
    regex = re.compile(regex[:-1])
    # Process citStr
    if "et al." in context_citStr:
      context_citStr = context_citStr.replace("et al.", "et al")
    # Process context
    if "et al." in context_lines:
      context_lines = context_lines.replace("et al.", "et al")
    query_lines = self.sentenceTokenizer.tokenize(context_lines)
    citationCount = 0
    for l in query_lines:
      obj = re.findall(regex, l)
      citationCount += len(obj)
    avgDensity = float(citationCount) / float(len(query_lines))
    return avgDensity

class dist:
  def __init__(self):
    self.sentenceTokenizer = PunktSentenceTokenizer()
    self.tools = tools()
    self.genericHeader = ['abstract',
                          'acknowledgements',
                          'background',
                          'categories and subject descriptors',
                          'conclusions',
                          'discussions',
                          'evaluation',
                          'general terms',
                          'introduction',
                          'keywords',
                          'method',
                          'references',
                          'related work',
                          'none'
                          ]

  def levenshtein(self, a, b):
    return distance.edit_distance(a, b)

  def levenshteinRatio(self, a, b):
    lensum = float(len(a) + len(b))
    if lensum == 0.0:
      return 1.0
    ldist = self.levenshtein(a,b)
    return (lensum - ldist) / lensum

  def jaccard(self, inputA, inputB):
    # Returns jaccard index. Smaller the more better
    a = inputA.lower()
    b = inputB.lower()
    return distance.jaccard_distance(set(a.split()), set(b.split()))

  def publishYear(self, cite_key):
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]

    citingYear = int(citing[1:3])
    citedYear = int(cited[1:3])

    if citingYear > 50:
      citingYear = 1900 + citingYear
    else:
      citingYear = 2000 + citingYear

    if citedYear > 50:
      citedYear = 1900 + citedYear
    else:
      citedYear = 2000 + citedYear
    return (citingYear-citedYear)

  def citSentLocation(self, cite_key, context_citStr, context, citingFile="/Users/lwheng/Downloads/fyp/parscitsectionxml500/"):
    citing = cite_key.split('==>')[0]
    citingFile = citingFile + citing + "-parscit-section.xml"
    vector = []
    if os.path.exists(citingFile):
      # Using context_citStr, first determine which is the citing sentence
      # We replace "et al." by "et al" so the sentence tokenizer doesn't split it up
      context_citStr = context_citStr.replace("et al.", "et al")
      context = context.replace("et al.", "et al")

      context_lines = self.sentenceTokenizer.tokenize(context)
      citSent = self.tools.searchTermInLines(context_citStr, context_lines)
      openfile = open(citingFile, 'r')
      data = openfile.read()
      openfile.close()
      dom = parseString(data)
      target = None
      bodyTexts = dom.getElementsByTagName('bodyText')
      regex = r"\<.*\>(.*)\<.*\>"
      tool = tools()

      minDistance = 314159265358979323846264338327950288419716939937510
      for i in range(len(bodyTexts)):
        b = bodyTexts[i]
        text = tool.normalize(b.toxml().replace("\n", " ").replace("- ", "").strip())
        obj = re.findall(regex, text)
        tempDist = self.jaccard(context_lines[citSent], text)
        if tempDist < minDistance:
          minDistance = tempDist
          target = b
      
      if target:
        searching = True
        sectionHeaderNode = None
        target = target.previousSibling
        while target:
          if target.nodeType == Node.ELEMENT_NODE:
            if target.nodeName == 'sectionHeader':
              sectionHeaderNode = target
              break
          target = target.previousSibling
        if target == None:
          for h in (self.genericHeader):
            vector.append(0)
          vector[-1] = 1 # Setting 'None' to 1
          return vector
        header = tool.normalize(sectionHeaderNode.attributes['genericHeader'].value)
        for h in (self.genericHeader):
          if header == h:
            vector.append(1)
          else:
            vector.append(0)
        return vector
        #return tool.normalize(sectionHeaderNode.attributes['genericHeader'].value)
      else:
        # Not found
        for h in (self.genericHeader):
          vector.append(0)
        vector[-1] = 1 # Setting 'None' to 1
        return vector
    else:
      # No section file
      for h in (self.genericHeader):
        vector.append(0)
      vector[-1] = 1 # Setting 'None' to 1
      return vector

class pickler:

  def __init__(self, paperTitles="/Users/lwheng/Downloads/fyp/paperTitles.pickle", paperAuthors="/Users/lwheng/Downloads/fyp/paperAuthors.pickle"):
    self.pickle_paperTitles = paperTitles
    self.pickle_paperAuthors = paperAuthors
    self.titles = self.fetchTitles()
    self.authors = self.fetchAuthors()

  def fetchTitle(self, paperID):
    return self.titles[paperID]

  def fetchTitles(self):
    tempTitle = {}
    tempTitle = pickle.load(open(self.pickle_paperTitles, "rb"))
    return tempTitle

  def fetchAuthors(self):
    tempAuthors = {}
    tempAuthors = pickle.load(open(self.pickle_paperAuthors, "rb"))
    return tempAuthors
