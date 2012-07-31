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
# 4. Publish Year
# 5. TF-IDF to determine most high-valued chunk in cited paper?
# 6. Title Overlap using Jaccard 
# 7. Authors Overlap

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
import pickle
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance

sentenceTokenizer = PunktSentenceTokenizer()

CHUNK_SIZE = 15
LAMBDA_AUTHOR_MATCH = 0.8
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

titles = {}
authors = {}
collection = ""

# Pickle files
pickle_paperTitles = "/Users/lwheng/Dropbox/fyp/code/paperTitles.pickle"
pickle_paperAuthors = "/Users/lwheng/Dropbox/fyp/code/paperAuthors.pickle"

def fetchCollection():
  print "Fetch Collection"

def fetchTitles():
  tempTitle = {}
  tempTitle = pickle.load(open(pickle_paperTitles, "rb"))
  return tempTitle

def fetchAuthors():
  tempAuthors = {}
  tempAuthors = pickle.load(open(pickle_paperAuthors, "rb"))
  return tempAuthors

def fetchContexts(cite_key):
  global titles
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]

  titleToMatch = titles[cited]

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
  minDistance = 314159265358979323846264338327950288419716939937510
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
      thisDistance = levenshtein(title, titleToMatch)
      if thisDistance < minDistance:
        minDistance = thisDistance
        bestIndex = i
  return citations[bestIndex]

def levenshtein(a, b):
  return distance.edit_distance(a,b)

def levenshteinRatio(a, b):
  lensum = float(len(a) + len(b))
  if lensum == 0.0:
    return 1.0
  ldist = levenshtein(a,b)
  return (lensum - ldist) / lensum

def jaccard(inputA, inputB):
  # Returns jaccard index. Smaller the more similar
  a = inputA.lower()
  b = inputB.lower()
  return distance.jaccard_distance(set(a.split()), set(b.split()))

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
  # regexBibYear = r"\s*year\s*=\s*\{?(\d{4})\}?\s*"
  # regexYear = r".*(\d{4}).*"

  # # A. Try to get citing publish year
  # rootDir = "/Users/lwheng/Downloads/fyp/annotations500/"
  # if os.path.exists(rootDir + citing + ".bib"):
  #   openfile = open(rootDir + citing + ".bib", "r")
  #   for l in openfile:
  #     matchObj = re.findall(regexBibYear, l)
  #     if matchObj:
  #       citingYear = int(matchObj[0])

  # # B. Try to get cited publish year
  # matchObj = re.findall(regexYear, context_citStr)
  # if matchObj:
  #   citedYear = int(matchObj[0])
  # else:
  #   if os.path.exists(rootDir + cited + ".bib"):
  #     openfile = open(rootDir + cited + ".bib", "r")
  #     for l in openfile:
  #       matchObj = re.findall(regexBibYear, l)
  #       if matchObj:
  #         citedYear = int(matchObj[0])
  return (citingYear,citedYear)

def titleOverlap(cite_key):
  global titles
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  return jaccard(titles[citing], titles[cited])

def authorOverlap(cite_key):
  global authors
  info = cite_key.split("==>")
  citing = info[0]
  cited = info[1]
  # Adapting the Jaccard idea
  matches = 0
  uniqueNames = len(authors[citing]) + len(authors[cited])
  for citingAuthor in authors[citing]:
    for citedAuthor in authors[cited]:
      ratio = levenshteinRatio(citingAuthor, citedAuthor)
      if ratio > LAMBDA_AUTHOR_MATCH:
        # A match
        matches += 1
        uniqueNames -= 1
  return float(matches) / float(uniqueNames)

def chunkWeight(chunk, doc_collection):
  weight = 0
  for t in chunk.tokens:
    weight += doc_collection.tf_idf(t, chunk)
  return weight

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
  maxChunkWeight = 0
  maxChunkIndex = 0
  for i in range(0, len(docs)):
    # TF-IDF to determine most high-valued chunk in cited paper?
    temp = chunkWeight(docs[i], docs_col)
    if temp > maxChunkWeight:
      maxChunkWeight = temp
      maxChunkIndex = i

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

  # TF-IDF to determine most high-valued chunk in cited paper?
  toprint = ""
  for t in docs[maxChunkIndex].tokens:
    toprint = toprint + " " + t
  print "### Chunk with max weight ###"
  print toprint
  print

  return results.index(min(results))


def citProv(cite_key):
  # Global scoping
  global context_dom
  global context_citStr
  global context_value
  global query_lines
  global query_display
  global query_tokens

  # 0. Set Up
  citation_dom = fetchContexts(cite_key)
  contexts = citation_dom.getElementsByTagName('context')
  for c in contexts:
    feature_vector = []

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
    feature_vector.append(feature_citDensity)   

    # 4. Publish Year
    feature_publishYear = publishYear(cite_key, context_citStr)
    feature_vector.append(feature_publishYear)

    # 6. Title Overlap
    feature_titleOverlap = titleOverlap(cite_key)
    feature_vector.append(feature_titleOverlap)

    # 7. Authors Overlap
    feature_authorOverlap = authorOverlap(cite_key)
    feature_vector.append(feature_authorOverlap)

    print cite_key + "==>" + str(feature_vector)

    # # 2. Using Cosine Similarity
    # resultIndex = cosineSimilarity(cite_key,context_value)
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
titles = fetchTitles()
authors = fetchAuthors()
collection = fetchCollection()
for l in startexperiment:
  info = l.split(",")
  cite_key = info[0]
  citProv(cite_key)
  sys.exit()












