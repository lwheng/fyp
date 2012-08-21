from xml.dom.minidom import parseString
import unicodedata
import nltk
import sys
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
    # regex = r"(((\w+)\s*,?\s*(et al.?)?|(\w+ and \w+))\s*,?\s*(\(?\s?\d{4}\s?\)?)|\[\s*(\w+)\s*\]|\[\s(\w+\d+)\s\]|[\[|\(]\s(\d+\s?,\s?)*(\d+)\s[\]|\)]|\(\s*[A-Z]\w+\s*\)|\[\s(\w+\s,?\s?)+\])"
    output = []
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
                          'related work'
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

  def citSentLocation(self, cite_key, context, citingFile="/Users/lwheng/Downloads/fyp/parscitsectionxml500/"):
    citing = cite_key.split('==>')[0]
    citingFile = citingFile + citing + "-parscit-section.xml"
    if os.path.exists(citingFile):
      openfile = open(citingFile, 'r')
      data = openfile.read()
      openfile.close()
      dom = parseString(data)
      variant = dom.getElementsByTagName('variant')[0]
      # Can consider using DOM Node.nodeType
      # And also to use DOM Node.previousSibling and Node.nextSibling, and nodeName = 'sectionHeader'/'subsectionHeader'
      return variant.childNodes
    else:
      return "No section file"

class pickler:
  # Pickle files
  # pickle_paperTitles = "/Users/lwheng/Downloads/fyp/paperTitles.pickle"
  # pickle_paperAuthors = "/Users/lwheng/Downloads/fyp/paperAuthors.pickle"
  # titles = {}
  # authors = {}
  # pickle_contextCollection = "/Users/lwheng/Downloads/fyp/contextCollection.pickle"

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
    # print "Loaded authors"
    return tempAuthors
