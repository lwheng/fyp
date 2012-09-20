from xml.dom.minidom import parseString
from xml.dom import Node
import unicodedata
import nltk
import re
import os
import math
import numpy as np
import cPickle as pickle
from sets import Set
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance
import sys

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
    if not type(text) == unicode:
      text = unicode(text, errors='ignore')
      return text
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore')

  def searchTermInLines(self, term, lines):
    for i in range(len(lines)):
      l = lines[i]
      if term in l:
        return i
    # Cannot find, return the mid of the chunk
    return int(len(lines)/2)

class weight:
  def __init__(self):
    self.sentenceTokenizer = PunktSentenceTokenizer()
    self.dist = dist()
    self.LAMBDA_AUTHOR_MATCH = 0.8
    reg = []
    reg.append(r"\(\s?(\d{1,3})\s?\)")
    reg.append(r"\(\s?(\d{4})\s?\)")
    reg.append(r"\(\s?(\d{4};?\s?)+\s?")
    reg.append(r"\[\s?(\d{1,3}\s?,?\s?)+\s?\]")
    reg.append(r"\[\s?([\w-],?\s?)+\s?\]")
    reg.append(r"et al\.?,?")
    #reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?,?\s?(\(?(\d{4})\)?)")
    #reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?(et al\.?)?\s?,?\s?(\(?(\d{4})\)?)")
    self.regex = ""
    for i in range(len(reg)):
      self.regex += reg[i] + "|"
    self.regex = re.compile(self.regex[:-1])

  def chunkAverageWeight(self, chunk, collection):
    tempWeight = 0
    if len(chunk.tokens) == 0:
      return 0
    for t in chunk.tokens:
      tempWeight += collection.tf_idf(t.lower(), chunk)
    return float(tempWeight) / float(len(chunk.tokens))
    
  def titleOverlap(self, cite_key, titles):
    return self.dist.jaccard(titles[cite_key['citing']], titles[cite_key['cited']])

  def titleOverlapCFS(self, title_citing, title_cited):
    return self.dist.jaccard(title_citing, title_cited)

  def authorOverlap(self, cite_key, authors):
    citing = cite_key['citing']
    cited = cite_key['cited']
    # Adapting the Jaccard idea
    matches = 0
    uniqueNames = len(authors[citing]) + len(authors[cited])
    for citingAuthor in authors[citing]:
      for citedAuthor in authors[cited]:
        ratio = self.dist.levenshteinRatio(citingAuthor, citedAuthor)
        if ratio > self.LAMBDA_AUTHOR_MATCH:
          matches += 1
          uniqueNames -= 1
    if uniqueNames == 0:
      return 1.0
    return float(matches) / float(uniqueNames)

  def authorOverlapCFS(self, authors_citing, authors_cited):
    matches = 0
    uniqueNames = len(authors_citing) + len(authors_cited)
    for citingAuthor in authors_citing:
      for citedAuthor in authors_cited:
        ratio = self.dist.levenshteinRatio(citingAuthor, citedAuthor)
        if ratio > self.LAMBDA_AUTHOR_MATCH:
          matches += 1
          uniqueNames -= 1
    if uniqueNames == 0:
      return 1.0
    return float(matches) / float(uniqueNames)

  def citDensity(self, context_lines, context_citStr):
    # Process citStr
    if "et al." in context_citStr:
      context_citStr = context_citStr.replace("et al.", "et al")
    # Process context
    if "et al." in context_lines:
      context_lines = context_lines.replace("et al.", "et al")
    query_lines = self.sentenceTokenizer.tokenize(context_lines)
    citationCount = 0
    for l in query_lines:
      obj = re.findall(self.regex, l)
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

  def masi(self, a, b):
    a = a.lower()
    b = b.lower()
    return distance.masi_distance(set(a.split()), set(b.split()))

  def publishYear(self, cite_key):
    citing = cite_key['citing']
    cited = cite_key['cited']

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

  def publishYearCFS(self, cite_key):
    citing = cite_key['citing']
    cited = cite_key['cited']

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

  def citSentLocation(self, cite_key, context_citStr, context, pathParscitSection):
    citing = cite_key['citing']
    citingFile = os.path.join(pathParscitSection, citing + "-parscit-section.xml")
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
        #text = tool.normalize(b.toxml().replace("\n", " ").replace("- ", "").strip())
        text = b.toxml().replace("\n", " ").replace("- ", "").strip()
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
        if sectionHeaderNode.attributes.has_key('genericHeader'):
          header = sectionHeaderNode.attributes['genericHeader'].value
        elif sectionHeaderNode.attributes.has_key('genericheader'):
          header = sectionHeaderNode.attributes['genericheader'].value
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

  def citSentLocationCFS(self, context_citStr, context, dom_citing_parscit_section):
    vector = []
    # Using context_citStr, first determine which is the citing sentence
    # We replace "et al." by "et al" so the sentence tokenizer doesn't split it up
    context_citStr = context_citStr.replace("et al.", "et al")
    context = context.replace("et al.", "et al")

    context_lines = self.sentenceTokenizer.tokenize(context)
    citSent = self.tools.searchTermInLines(context_citStr, context_lines)
    dom = dom_citing_parscit_section
    target = None
    bodyTexts = dom.getElementsByTagName('bodyText')
    regex = r"\<.*\>(.*)\<.*\>"
    tool = tools()

    minDistance = 314159265358979323846264338327950288419716939937510
    for i in range(len(bodyTexts)):
      b = bodyTexts[i]
      text = b.toxml().replace("\n", " ").replace("- ", "").strip()
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
      if sectionHeaderNode.attributes.has_key('genericHeader'):
        header = sectionHeaderNode.attributes['genericHeader'].value
      elif sectionHeaderNode.attributes.has_key('genericheader'):
        header = sectionHeaderNode.attributes['genericheader'].value
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

class pickler:
  def __init__(self):
    config = pickle.load(open("Config.pickle", "r"))
    self.pathRoot = config[0]
    self.pathCode = config[1]
    self.pathParscit = os.path.join(self.pathRoot, "parscitxml")
    self.pathParscitSection = os.path.join(self.pathRoot, "parscitsectionxml")
    self.pathPDFBox = os.path.join(self.pathRoot, "pdfbox-0.72")

    # Pickles
    self.pathAnnotations = os.path.join(self.pathRoot, "Annotations.pickle")
    self.pathAuthors = os.path.join(self.pathRoot, "Authors.pickle")
    self.pathDataset = os.path.join(self.pathRoot, "Dataset.pickle")
    self.pathDatasetTBA = os.path.join(self.pathRoot, "DatasetTBA.pickle")
    self.pathDatasetTBACFS = os.path.join(self.pathRoot, "DatasetTBACFS.pickle")
    self.pathDatasetTBA_keys = os.path.join(self.pathRoot, "DatasetTBA_keys.pickle")
    self.pathExperiment = os.path.join(self.pathRoot, "Experiment.pickle")
    self.pathForAnnotation = os.path.join(self.pathRoot, "For_Annotation.pickle")
    self.pathModel = os.path.join(self.pathRoot, "Model.pickle")
    self.pathModelCFS = os.path.join(self.pathRoot, "ModelCFS.pickle")
    self.pathRaw = os.path.join(self.pathRoot, "Raw.pickle")
    self.pathTarget = os.path.join(self.pathRoot, "Target.pickle")
    self.pathTitles = os.path.join(self.pathRoot, "Titles.pickle")

  def loadPickle(self, filename):
    temp = pickle.load(open(filename, "rb"))
    return temp

  def dumpPickle(self, data, filename):
    pickle.dump(data, open(filename+".pickle", "wb"))

class dataset_tools:
  def __init__(self, dist, nltk_Tools, pickler, tools):
    self.dist = dist
    self.nltk_Tools = nltk_Tools
    self.parscitPath = pickler.pathParscit
    self.parscitSectionPath = pickler.pathParscitSection
    self.tools = tools
    self.LAMBDA_ANNOTATION_MATCH = 0.5

  def fetchExperiment(self, raw):
    experiment = []
    for k in raw.keys():
      record = {}
      info = k.split("==>")
      record['citing'] = info[0]
      record['cited'] = info[1]
      experiment.append(record)
    return experiment

  def prepContexts(self, dist, tools, titles, cite_key):
    citing = cite_key['citing']
    cited = cite_key['cited']
    titleToMatch = titles[cited]

    citingFile = os.path.join(self.parscitPath, citing+"-parscit.xml")
    openciting = open(citingFile,"r")
    data = openciting.read()
    openciting.close()
    dom = tools.parseXML(data)
    citations = dom.getElementsByTagName('citation')
    tags = ["title", "note", "booktitle", "journal", "tech", "author"]
    titleTag = []
    index = 0
    bestIndex = -1
    minDistance = 314159265358979323846264338327950288419716939937510
    for i in range(len(citations)):
      c = citations[i]
      #valid = c.getAttribute('valid')
      #if valid == "true":
      titleTag = []
      for index in range(len(tags)):
        titleTag = c.getElementsByTagName(tags[index])
        if titleTag:
          break
      if titleTag == [] or titleTag[0].firstChild == None:
        continue
      title = titleTag[0].firstChild.data
      if not type(title) == unicode:
        title = tools.normalize(title)
      if re.search("Computational Linguistics,$", title):
        title = title.replace("Computational Linguistics,", "")
      levenshteinDistance = dist.levenshtein(title.lower(), titleToMatch.lower())
      masiDistance = dist.masi(title, titleToMatch)
      thisDistance = levenshteinDistance*masiDistance
      if thisDistance < minDistance:
        minDistance = thisDistance
        bestIndex = i
    if bestIndex == -1:
      return None
    return citations[bestIndex]

  def prepContextsCFS(self, dist, tools, title_citing, title_cited, dom_citing_parscit):
    titleToMatch = title_cited
    dom = dom_citing_parscit

    citations = dom.getElementsByTagName('citation')
    tags = ["title", "note", "booktitle", "journal", "tech", "author"]
    titleTag = []
    index = 0
    bestIndex = -1
    minDistance = 314159265358979323846264338327950288419716939937510
    for i in range(len(citations)):
      c = citations[i]
      titleTag = []
      for index in range(len(tags)):
        titleTag = c.getElementsByTagName(tags[index])
        if titleTag:
          break
      if titleTag == [] or titleTag[0].firstChild == None:
        continue
      title = titleTag[0].firstChild.data
      if not type(title) == unicode:
        title = tools.normalize(title)
      if re.search("Computational Linguistics,$", title):
        title = title.replace("Computational Linguistics,", "")
      levenshteinDistance = dist.levenshtein(title.lower(), titleToMatch.lower())
      masiDistance = dist.masi(title, titleToMatch)
      thisDistance = levenshteinDistance*masiDistance
      if thisDistance < minDistance:
        minDistance = thisDistance
        bestIndex = i
    if bestIndex == -1:
      return None
    return citations[bestIndex]

  def prepRaw(self, authors, experiment, titles):
    raw = {}
    for e in experiment:
      record = {}
      dom = self.prepContexts(self.dist, self.tools, titles, e)
      if dom:
        contexts = dom.getElementsByTagName('context')
        if len(contexts) > 0:
          record['citing'] = {'authors':authors[e['citing']], 'title':titles[e['citing']]}
          record['cited'] = {'authors':authors[e['cited']], 'title':titles[e['cited']]}
          record['contexts'] = contexts
          raw[str(e['citing']+"==>"+e['cited'])] = record
    return raw

  def prepDataset(self, run, raw, experiment, annotations):
    # Raw has 500
    # Experiment has 500
    # Annotations has 757
    forannotation = []
    dataset = []
    keys = []
    targets = []
    indexAnnotations = 0
    indexInstances = 0

    for e in experiment:
      contexts = raw[e['citing']+"==>"+e['cited']]['contexts']
      context_list = []
      for c in contexts:
        value = self.tools.normalize(c.firstChild.data).lower()
        context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
      citing_col = self.nltk_Tools.nltkTextCollection(context_list)
      for c in contexts:
        currentAnnotation = annotations[indexAnnotations]
        indexAnnotations += 1
        x = run.extractFeatures(e, c, citing_col)
        forannotation.append((e, c))
        instances = []
        featuresLessCosSim = x[:-1]
        for i in x[-1]:
          target = self.prepTarget(currentAnnotation, i[0])
          targets.append(target)
          temp = featuresLessCosSim[:]
          # i[1][1] is chunkAvgWeight
          temp.append(i[1][1])
          # i[1][0] is cosineSim
          temp.append(i[1][0].item())
          instances.append(temp)
          keys.append(e)
        dataset.extend(instances)
    X = np.asarray(dataset)
    targets = np.asarray(targets)
    return (forannotation, keys, X, targets)

  def prepDatasetCFS(self, run, raw, experiment):
    forannotation = []
    dataset = []
    keys = []
    for e in experiment:
      contexts = raw[e['citing']+"==>"+e['cited']]['contexts']
      context_list = []
      for c in contexts:
        value = c.firstChild.data.lower()
        value = unicodedata.normalize('NFKD', value).encode('utf-8','ignore')
        context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
      citing_col = self.nltk_Tools.nltkTextCollection(context_list)
      for c in contexts:
        x = run.extractFeaturesCFS(e, c, citing_col)
        forannotation.append((e, c))
        keys.append(e)
        dataset.append(x)
    X = np.asarray(dataset)
    return (forannotation, keys, X)

  def prepDatasetCFS_v2(self, run, raw, experiment):
    forannotation = []
    dataset = []
    keys = []
    for e in experiment:
      contexts = raw[e['citing']+"==>"+e['cited']]['contexts']
      context_list = []
      for c in contexts:
        value = c.firstChild.data.lower()
        value = unicodedata.normalize('NFKD', value).encode('utf-8','ignore')
        context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
      citing_col = self.nltk_Tools.nltkTextCollection(context_list)
      for c in contexts:
        x = run.extractFeaturesCFS(e, c, citing_col)
        forannotation.append((e, c))
        keys.append(e)
        dataset.append(x)
    X = np.asarray(dataset)
    return (forannotation, keys, X)

  def prepAnnotations(self, annotationFile):
    regex = r"\#(\d{3})\s+(.*)==>(.*),(.*)"
    target = []
    for l in open(annotationFile):
      l = l.strip()
      obj = re.findall(regex, l)
      info = obj[0]
      index = int(info[0])
      cite_key = {'citing':info[1], 'cited':info[2]}
      annotation = info[3]
      target.append((index, cite_key, annotation))
    temp = []
    for t in target:
      temp.append(t[2])
    y = np.asarray(temp)
    return y

  def prepModel(self, classifier, dataset, target):
    classifier.fit(dataset, target)
    return classifier

  def prepTarget(self, annotation, chunk):
    # General - 0
    # Specific - Yes - 1
    # Specific - No - 2
    # Undetermined - 3
    if annotation == "-":
      return 0
    elif annotation == "?":
      return 3
    else:
      ranges = []
      temp = annotation.split("!")
      for t in temp:
        if t:
          top = t.split("-")[0]
          bottom = t.split("-")[1]
          ranges.append((int(top),int(bottom)))

      temp = chunk.split("-")
      top = int(temp[0])
      bottom = int(temp[1])
      chunkRange = range(top, bottom+1)
      chunkRange_set = Set(chunkRange)
      for r in ranges:
        testRange = range(r[0], r[1]+1)
        testRange_set = Set(testRange)
        intersect_set = testRange_set & chunkRange_set
        if float(len(list(intersect_set)))/float(len(list(testRange))) > self.LAMBDA_ANNOTATION_MATCH:
          return 1
      return 2

class classifier:
  def __init__(self, classifier):
    self.data = None
    self.target = None
    # Specify what classifier to use here
    self.clf = classifer

  def prepClassifier(self, data, target):
    # Data: Observations
    # Target: Known classifications
    self.clf.fit(data, target)
    return self.clf

  def predict(self, observation):
    # Takes in an observation and returns a prediction
    return self.clf.predict(observation)

