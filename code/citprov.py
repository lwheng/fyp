import Utils

class citprov:
	def __init__(self):
		self.pickler = Utils.pickler()
		self.dist = Utils.dist()
		self.LAMBDA_AUTHOR_MATCH = 0.8

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

	def fetchContexts(self, cite_key):
		info = cite_key.split("==>")
		citing = info[0]
		cited = info[1]

		titleToMatch = self.titles[cited]

		citingFile = "/Users/lwheng/Downloads/fyp/parscitxml500/" + citing + "-parscit.xml"
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