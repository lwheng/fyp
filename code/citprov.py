#!/opt/local/bin/python
# -*- coding: utf-8 -*-

# To determine the type of the citation.
# If specific, we wish to determine the location of the cited information

# 1. Using Citation Density 
# 2. Using Cosine Similarity
# 3. Using Sentence Tokenizer

# 4. Location of citing sentence
# 5. Popularity
# 6. Density 
# 7. AvgDens

# 8. Cue Words 
# 9. POS Tagging

from xml.dom.minidom import parseString
import unicodedata
import nltk
# import HTMLParser
import sys
import re
import math
import numpy
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
import Levenshtein

cite_key = "W03-0415==>P99-1016"
contextDemo = """<context citStr=\"Caraballo (1999)\" endWordPosition=\"593\" position=\"3648\" startWordPosition=\"592\">. In Section 4, we show how correctly extracted relationships can be used as seed-cases to extract several more relationships, thus improving recall; this work shares some similarities with that of Caraballo (1999). In Section 5 we show that combining the techniques of Section 3 and Section 4 improves both precision and recall. Section 6 demonstrates that 1Another possible view is that hyponymy should only re</context>"""
citStr = ""

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

def citDensity(inputText):

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

  numOfCitations = 0

  for l in inputText:
    obj = re.findall(regex, l)
    numOfCitations += len(obj)

  return (float(numOfCitations) / float(len(inputText)), numOfCitations)


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

    # 3. Using Sentence Tokenizer
    query_lines = sentenceTokenizer.tokenize(context_value)

    # 1. Using Citation Density
    (citDensityValue, numOfCitations) = citDensity(query_lines)
    print cite_key + " ==> " + str((citDensityValue, numOfCitations))

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












