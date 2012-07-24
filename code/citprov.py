#!/opt/local/bin/python
# -*- coding: utf-8 -*-

# To determine the type of the citation.
# If specific, we wish to determine the location of the cited information

# 1. Using Citation Density 
    # Popularity: no. of references cited in the same sentence
    # Density:    no. of difference references in citing/neighbour sentences
    # AvgDens:    average of Density among neighbour sentences surrounding the citation sentence
# 2. Using Cosine Similarity
# 3. Using Sentence Tokenizer
# 4. Publish Year?

# x. Location of citing sentence
# x+1. Cue Words 
# x+2. POS Tagging

from xml.dom.minidom import parseString
import unicodedata
import nltk
# import HTMLParser
import sys
import re
import os
import math
import numpy
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
import Levenshtein

sentenceTokenizer = PunktSentenceTokenizer()

CHUNK_SIZE = 15
citationTypes = ['General', 'Specific', 'Undetermined']

punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"

query_display = ""
query_tokens = []
query_lines = []
query_col = []
query_fd = []

context_dom = ""
context_citStr = ""
context_value = ""

docs_col = []

vocab = []

title = {}

def fetchTitles():
  tempTitle = {}
  titleFile = "/Users/lwheng/Dropbox/fyp/annotation/paperTitles.txt"
  opentitlefile = open(titleFile, "r")
  for l in opentitlefile:
    line = l.strip()
    info = line.split("==>")
    tempTitle[info[0]] = info[1]
  opentitlefile.close()
  return tempTitle

def fetchContexts(cite_key):
  global title
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]

  titleToMatch = title[cited]

  citingFile = "/Users/lwheng/Downloads/fyp/annotations500/" + citing + "-parscit.xml"
  openciting = open(citingFile,"r")
  data = openciting.read()
  openciting.close()
  dom = parseString(data)
  citations = dom.getElementsByTagName('citation')
  tags = ["title", "note", "booktitle", "journal"]
  titleTag = []
  index = 0
  bestIndex = 0
  maxRatio = 0
  for i in range(len(citations)):
    c = citations[i]
    valid = c.getAttribute('valid')
    if valid == "true":
      titleTag = []
      index = 0
      while titleTag == []:
        titleTag = c.getElementsByTagName(tags[index])
        index += 1
      title = titleTag[0].firstChild.data
      title = unicodedata.normalize('NFKD', title).encode('ascii','ignore')
      ratio = Levenshtein.ratio(str(title), str(titleToMatch))
      if ratio > maxRatio:
        maxRatio = ratio
        bestIndex = i
  return citations[bestIndex]

def citDensity(context_lines, context_citStr):

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

  # Tuple: (CitingSetence?, Density)
  output = []

  # Process citStr
  if "et al." in context_citStr:
    context_citStr = context_citStr.replace("et al.", "et al")
  # Process context
  if "et al." in context_lines:
    context_lines = context_lines.replace("et al.", "et al")

  # 3. Using Sentence Tokenizer
  query_lines = sentenceTokenizer.tokenize(context_lines)

  checker = False
  for l in query_lines:
    obj = re.findall(regex, l)
    if context_citStr in l:
      output.append((True, len(obj)))
      checker = True
    else:
      output.append((False, len(obj)))

  density = 0
  avgdensity = 0
  for t in output:
    if t[0]:
      density += t[1]
    else:
      avgdensity += t[1]
  return (density, float(avgdensity)/float(len(output)))

def publishYear(cite_key, context_citStr):
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]

  citingYear = 0
  citedYear = 0
  regexBibYear = r"\s*year\s*=\s*\{?(\d{4})\}?\s*"
  regexYear = r".*(\d{4}).*"

  # A. Try to get citing publish year
  rootDir = "/Users/lwheng/Downloads/fyp/annotations500/"
  if os.path.exists(rootDir + citing + ".bib"):
    openfile = open(rootDir + citing + ".bib", "r")
    for l in openfile:
      matchObj = re.findall(regexBibYear, l)
      if matchObj:
        citingYear = int(matchObj[0])

  # B. Try to get cited publish year
  matchObj = re.findall(regexYear, context_citStr)
  if matchObj:
    citedYear = int(matchObj[0])
  else:
    if os.path.exists(rootDir + cited + ".bib"):
      openfile = open(rootDir + cited + ".bib", "r")
      for l in openfile:
        matchObj = re.findall(regexBibYear, l)
        if matchObj:
          citedYear = int(matchObj[0])
  return (citingYear,citedYear)

def cosineSimilarity(cite_key, context):
  global query_tokens
  global query_display
  global domain
  global docs
  global CHUNK_SIZE
  global vocab

  # Citing Paper
  query_fd = nltk.FreqDist(query_tokens)
  query_col = nltk.TextCollection([nltk.Text(query_tokens)])

  # Cited Paper
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  citedpaper = "/Users/lwheng/Downloads/fyp/pdfbox-0.72/" + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
  domain = []
  try:
    openfile = open(citedpaper,"r")
    for l in openfile:
      domain.append(l.strip())
    openfile.close()
  except IOError as e:
    print e
  docs = []
  for i in xrange(0, len(domain), CHUNK_SIZE/2):
    sublist = domain[i:i+CHUNK_SIZE]
    temp = ""
    for s in sublist:
      temp = temp + " " + s
    docs.append(nltk.Text(nltk.word_tokenize(temp)))
  docs_col = nltk.TextCollection(docs)

  # Vocab
  vocab = list(set(query_col) | set(docs_col))
  vocab = map(lambda x: x.lower(), vocab)
  vocab = [w for w in vocab if not w in stopwords.words('english')]
  vocab = [w for w in vocab if not w in punctuation]

  # Prep Vectors
  results = []
  for i in range(0, len(docs)):
    u = []
    v = []
    fd_doc_current = nltk.FreqDist(docs[i])
    temp_query = map(lambda x: x.lower(), query_tokens)
    temp_query = [w for w in temp_query if not w in stopwords.words('english')]
    temp_query = [w for w in temp_query if not w in punctuation]
    temp_doc = map(lambda x: x.lower(), docs[i])
    temp_doc = [w for w in temp_doc if not w in stopwords.words('english')]
    temp_doc = [w for w in temp_doc if not w in punctuation]
    for term in vocab:
      if term in temp_query:
        u.append(docs_col.tf_idf(term, temp_doc))
      else:
        u.append(0)
      if term in temp_doc:
        v.append(docs_col.tf_idf(term, temp_doc))
      else:
        v.append(0)
    if math.sqrt(numpy.dot(u, u)) == 0.0:
      results.append(1000)
    else:
      r = nltk.cluster.util.cosine_distance(u,v)
      results.append(r)
  return results.index(min(results))


def citProv(cite_key):
  # Global scoping
  global context_dom
  global context_citStr
  global context_value
  global query_lines
  global query_display
  global query_tokens
  global title

  # 0. Set Up
  title = fetchTitles()
  citation_dom = fetchContexts(cite_key)
  contexts = citation_dom.getElementsByTagName('context')
  for c in contexts:
    context_citStr = c.getAttribute('citStr')
    context_citStr = unicodedata.normalize('NFKD', context_citStr).encode('ascii','ignore')
    context_value = c.firstChild.data
    context_value = unicodedata.normalize('NFKD', context_value).encode('ascii','ignore')

    query_lines = context_value
    query_tokens = nltk.word_tokenize(context_value)
    query_display = ""
    for t in query_tokens:
      query_display = query_display + " " + t

    # 1. Using Citation Density
    feature_citDensity = citDensity(query_lines, context_citStr)

    # 4. Publish Year
    feature_publishYear = publishYear(cite_key, context_citStr)
    print cite_key + " ==> " + str(feature_citDensity) + "   " + str(feature_publishYear)
    # sys.exit()

    # # 2. Using Cosine Similarity
    # resultIndex = cosineSimilarity(cite_key,contextDemo)
    # print "### Query ###"
    # print query_display
    # print 
    # toprint = ""
    # for t in docs[resultIndex].tokens:
    #   toprint = toprint + " " + t
    # print "### Guess ###"
    # print toprint
    # print

experiment50 = "/Users/lwheng/Dropbox/fyp/annotation/annotations50.txt"
startexperiment = open(experiment50,"r")
for l in startexperiment:
  info = l.split(",")
  cite_key = info[0]
  citProv(cite_key)
  # sys.exit()












