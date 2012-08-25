import Utils
import math
import numpy

class citprov:
  def __init__(self):
    self.pickler = Utils.pickler()
    self.dist = Utils.dist()
    self.tools = Utils.tools()
    self.weight = Utils.weight()
    self.nltk_Tools = Utils.nltk_tools()

    self.stopwords = self.nltk_Tools.nltkStopwords()
    self.LAMBDA_AUTHOR_MATCH = 0.8
    self.CHUNK_SIZE = 15
    self.punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"
    self.citationTypes = ['General', 'Specific', 'Undetermined']
    self.genericHeader = [ 'abstract',
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

  def sayHello(self):
    return "Hello, world! I'm Citprov!"

  def titleOverlap(self, cite_key, titles):
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]
    return self.dist.jaccard(titles[citing], titles[cited])

  def authorOverlap(self, cite_key, authors):
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]
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

  def fetchContexts(self, cite_key, titles, citingFile="/Users/lwheng/Downloads/fyp/parscitxml500/"):
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]

    titleToMatch = titles[cited]

    # citingFile = "/Users/lwheng/Downloads/fyp/parscitxml500/" + citing + "-parscit.xml"
    citingFile = citingFile + citing + "-parscit.xml"
    openciting = open(citingFile,"r")
    data = openciting.read()
    openciting.close()
    dom = self.tools.parseXML(data)
    citations = dom.getElementsByTagName('citation')
    tags = ["title", "note", "booktitle", "journal"]
    titleTag = []
    index = 0
    bestIndex = -1
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
        title = self.tools.normalize(title)
        thisDistance = self.dist.levenshtein(title, titleToMatch)
        if thisDistance < minDistance:
          minDistance = thisDistance
          bestIndex = i
    if bestIndex == -1:
      return None
    return citations[bestIndex]

  def cosineSimilarity(self, cite_key, query_tokens, query_col, citedpaper="/Users/lwheng/Downloads/fyp/pdfbox-0.72/"):
    # Cited Paper
    info = cite_key.split("==>")
    citing = info[0]
    cited = info[1]
    # citedpaper = "/Users/lwheng/Downloads/fyp/pdfbox-0.72/" + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
    citedpaper = citedpaper + cited[0] + "/" + cited[0:3] + "/" + cited + ".txt"
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
      # 7.1 Cited Chunk's Average TF-IDF Weight
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
      if math.sqrt(numpy.dot(u, u)) == 0.0:
        results.append((0.0, 0.0))
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

  def citProv(self, cite_key):
    info = cite_key.split('==>')
    citing = info[0]
    cited = info[1]

    citation_dom = self.fetchContexts(cite_key, self.pickler.titles)
    if citation_dom:
      contexts = citation_dom.getElementsByTagName('context')
      # What if no context? --> According to ParsCit, citation is valid, but has no contexts
    else:
      # No citation
      return [cite_key, None, '-']

    # Prep citing_col
    context_list = []
    for c in contexts:
      value = c.firstChild.data
      value = self.tools.normalize(value)
      value = value.lower()
      tempText = self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value))
      context_list.append(tempText)
    citing_col = self.nltk_Tools.nltkTextCollection(context_list)

    data = []

    for c in contexts:
      feature_vector = []
      feature_vector.append(cite_key)
      feature_vector.append(c)

      context_citStr = c.getAttribute('citStr')
      context_citStr = self.tools.normalize(context_citStr)
      context_value = c.firstChild.data
      context_value = self.tools.normalize(context_value)

      query_lines = context_value
      query_tokens = self.nltk_Tools.nltkWordTokenize(context_value.lower())
      query_col = self.nltk_Tools.nltkTextCollection([self.nltk_Tools.nltkText(query_tokens)])
      query_display = ""
      for t in query_tokens:
        query_display = query_display + " " + t

      # 1. Using Citation Density
      feature_citDensity = self.weight.citDensity(query_lines, context_citStr)
      feature_vector.append(feature_citDensity)  

      # 2. Publishing Year Difference
      feature_publishYear = self.dist.publishYear(cite_key)
      feature_vector.append(feature_publishYear)

      # 3. Title Overlap
      feature_titleOverlap = self.titleOverlap(cite_key, self.pickler.titles)
      feature_vector.append(feature_titleOverlap)

      # 4. Authors Overlap
      feature_authorOverlap = self.authorOverlap(cite_key, self.pickler.authors)
      feature_vector.append(feature_authorOverlap)

      # 5. Context's Average TF-IDF Weight
      feature_queryWeight = self.weight.chunkAverageWeight(self.nltk_Tools.nltkText(query_tokens), citing_col)
      feature_vector.append(feature_queryWeight)

      # 6. Location Of Citing Sentence
      feature_locationCitingSent = self.dist.citSentLocation(cite_key, context_citStr, context_value)
      feature_vector.append(feature_locationCitingSent)

      # 7. Cosine Similarity + 7.1 Cited Chunk's Average TF-IDF Weight
      # Note: For n chunks in cited paper we perform cosineSimilarity,
      # so we have n results
      # We skip this part for now
      #feature_cosineSimilarity = self.cosineSimilarity(cite_key, query_tokens, query_col)
      #feature_vector.append(feature_cosineSimilarity)

      #print cite_key + " : " + str(feature_vector[0:-1])
      # for i in feature_cosineSimilarity:
      #   display = feature_vector[0:-1]
      #   display.append(i[1][0])
      #   display.append(i[1][1])
      #   print cite_key + " : " + str(i[0]) + " : " + str(display)
        # print i[0] + "\t" + str(i[1])

      data.append(feature_vector)
    return data
