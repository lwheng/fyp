import Utils
import math

class citprov:
	def __init__(self):
		self.pickler = Utils.pickler()
		self.dist = Utils.dist()
		self.tools = Utils.tools()
		self.weight = Utils.weight()
		self.nltk_Tools = Utils.nltk_Tools()

		self.stopwords = self.nltk_Tools.nltkStopwords()
		self.LAMBDA_AUTHOR_MATCH = 0.8
		self.CHUNK_SIZE = 15
		self.punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"

	def titleOverlap(self, cite_key, titles):
		info = cite_key.split("==>")
		citing = info[0]
		cited = info[1]
		return self.dist.jaccard(titles[citing], titles[cited])

	def authorOverlap(self, cite_key):
		authors = self.pickler.authors

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
		dom = self.tools.parseString(data)
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
				title = self.tools.normalize('NFKD', title).encode('ascii','ignore')
				thisDistance = self.dist.levenshtein(title, titleToMatch)
				if thisDistance < minDistance:
					minDistance = thisDistance
					bestIndex = i
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
			text = self.nltk_Tools.nltkText(self.nltk_Tools.wordTokenize(temp.lower()))
			docs.append(text)
			docs_display.append((str(i) + "-" + str(i+CHUNK_SIZE), text))
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