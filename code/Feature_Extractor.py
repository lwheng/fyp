import Utils
import math
import numpy as np

class extractor:
  def __init__(self, dist, nltk_Tools, pickler, tools, weight):
    self.dist = dist
    self.nltk_Tools = nltk_Tools
    self.pickler = pickler
    self.tools = tools
    self.weight = weight

    self.titles = self.pickler.loadPickle(self.pickler.pathTitles)
    self.authors = self.pickler.loadPickle(self.pickler.pathAuthors)
    self.stopwords = self.nltk_Tools.nltkStopwords()
    self.LAMBDA_AUTHOR_MATCH = 0.8
    self.CHUNK_SIZE = 15
    self.punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"
    self.citationTypes = ['General', 'Specific', 'Undetermined']

  def sayHello(self):
    return "Hello, world! I'm Citprov! If you can see this, then you are able to call my methods"

  def cosineSimilarity(self, cite_key, query_tokens, query_col, pathPDFBox):
    citing = cite_key['citing']
    cited = cite_key['cited']
    citedpaper = pathPDFBox + "/" + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
    domain = []
    try:
      openfile = open(citedpaper,"r")
      for l in openfile:
        domain.append(l.strip())
      openfile.close()
    except IOError as e:
      print e
    docs = []
    docs_display = []
    for i in xrange(0, len(domain), self.CHUNK_SIZE/2):
      sublist = domain[i:i+self.CHUNK_SIZE]
      temp = ""
      for s in sublist:
        temp = temp + " " + s
      text = self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(temp.lower()))
      docs.append(text)
      docs_display.append((str(i) + "-" + str(i+self.CHUNK_SIZE), text))
    docs_col = self.nltk_Tools.nltkTextCollection(docs)

    # Vocab
    vocab = list(set(query_col) | set(docs_col))
    vocab = map(lambda x: x.lower(), vocab)
    vocab = [w for w in vocab if not w in self.stopwords]
    vocab = [w for w in vocab if not w in self.punctuation]

    # Prep Vectors
    results = []
    for i in range(0, len(docs)):
      # Cited Chunk's Average TF-IDF Weight
      chunkAvgWeight = self.weight.chunkAverageWeight(docs[i], docs_col)

      u = []
      v = []
      # fd_doc_current = nltk.FreqDist(docs[i])
      temp_query = map(lambda x: x.lower(), query_tokens)
      temp_query = [w for w in temp_query if not w in self.stopwords]
      temp_query = [w for w in temp_query if not w in self.punctuation]
      temp_doc = map(lambda x: x.lower(), docs[i])
      temp_doc = [w for w in temp_doc if not w in self.stopwords]
      temp_doc = [w for w in temp_doc if not w in self.punctuation]
      for term in vocab:
        if term in temp_query:
          try:
            u.append(docs_col.tf_idf(term, temp_doc))
          except:
            u.append(0)
        else:
          u.append(0)
        if term in temp_doc:
          v.append(docs_col.tf_idf(term, temp_doc))
        else:
          v.append(0)
      if math.sqrt(np.dot(u, u)) == 0.0:
        results.append((np.float64(0.0), 0.0))
      else:
        r = self.nltk_Tools.nltkCosineDistance(u,v)
        # results.append(r)
        results.append((r,chunkAvgWeight))
    # total = sum(results)
    feature = []
    for i in range(len(results)):
      feature.append((docs_display[i][0],results[i]))
      # feature.append((docs_display[i][0],float(results[i])/float(total)*100))
    return feature

  def extractFeatures(self, cite_key, context, citing_col):
    citing = cite_key['citing']
    cited = cite_key['cited']

    # Context is ONE context
    citStr = context.getAttribute('citStr')
    citStr = self.tools.normalize(citStr)
    query = context.firstChild.data
    query = self.tools.normalize(query)

    query_tokens = self.nltk_Tools.nltkWordTokenize(query.lower())
    query_text = self.nltk_Tools.nltkText(query_tokens)
    query_col = self.nltk_Tools.nltkTextCollection([query_text])

    # Extract Features
    x = []

    # Citation Density
    feature_citDensity = self.weight.citDensity(query, citStr)
    x.append(feature_citDensity)

    # Publishing Year Difference
    feature_publishYear = self.dist.publishYear(cite_key)
    x.append(feature_publishYear)

    # Title Overlap
    feature_titleOverlap = self.weight.titleOverlap(cite_key, self.titles)
    x.append(feature_titleOverlap)

    # Authors Overlap
    feature_authorOverlap = self.weight.authorOverlap(cite_key, self.authors)
    x.append(feature_authorOverlap)

    # Context's Average TF-IDF Weight
    feature_queryWeight = self.weight.chunkAverageWeight(query_text, citing_col)
    x.append(feature_queryWeight)

    # Location of Citing Sentence
    feature_locationCitSent = self.dist.citSentLocation(cite_key, citStr, query, self.pickler.pathParscitSection)
    x.extend(feature_locationCitSent)

    # Cosine Similarity
    feature_cosineSimilarity = self.cosineSimilarity(cite_key, query_tokens, query_col, self.pickler.pathPDFBox)
    x.append(feature_cosineSimilarity)
    return x

  def extractFeaturesRaw(self, context, citing_col, dom_citing_parscit_section, title_citing, title_cited, authors_citing, authors_cited):
    # Context is ONE context
    citStr = context.getAttribute('citStr')
    query = context.firstChild.data

    query_tokens = self.nltk_Tools.nltkWordTokenize(query.lower())
    query_text = self.nltk_Tools.nltkText(query_tokens)
    query_col = self.nltk_Tools.nltkTextCollection([query_text])

    # Extract Features
    x = []

    # Citation Density
    feature_citDensity = self.weight.citDensity(query, citStr)
    x.append(feature_citDensity)

    # Publishing Year Difference
    #feature_publishYear = self.dist.publishYear(cite_key)
    #x.append(feature_publishYear)

    # Title Overlap
    feature_titleOverlap = self.weight.titleOverlapRaw(title_citing, title_cited)
    x.append(feature_titleOverlap)

    # Authors Overlap
    feature_authorOverlap = self.weight.authorOverlapRaw(authors_citing, authors_cited)
    x.append(feature_authorOverlap)

    # Context's Average TF-IDF Weight
    feature_queryWeight = self.weight.chunkAverageWeight(query_text, citing_col)
    x.append(feature_queryWeight)

    # Location of Citing Sentence
    feature_locationCitSent = self.dist.citSentLocationRaw(citStr, query, dom_citing_parscit_section)
    x.extend(feature_locationCitSent)

    # Cosine Similarity
    #feature_cosineSimilarity = self.cosineSimilarity(cite_key, query_tokens, query_col, self.pickler.pathPDFBox)
    #x.append(feature_cosineSimilarity)
    return x
