from xml.dom.minidom import parseString
from xml.dom import Node
import unicodedata
import nltk
import re
import os
import math
import decimal
import numpy as np
import cPickle as pickle
from sets import Set
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.metrics import distance
from nltk import SnowballStemmer
from sklearn.linear_model import LogisticRegression
import sys
import operator

class printer:
  def line_printer(self, length, character):
    output = ""
    for i in range(length):
      output += character
    return output

class nltk_tools:
  def __init__(self):
    self.stemmer = SnowballStemmer("english")

  def nltk_word_tokenize(self, text):
    return nltk.word_tokenize(text)

  def nltk_text(self, tokens):
    return nltk.Text(tokens)

  def nltk_text_collection(self, documents):
    return nltk.TextCollection(documents)

  def nltk_stopwords(self):
    return stopwords.words('english')

  def nltk_cosine_distance(self, u, v):
    # Closer to 1 the better
    return nltk.cluster.util.cosine_distance(u,v)

  def nltk_stemmer(self, input_string):
    return self.stemmer.stem(input_string)

  def nltk_pos(self, text):
    return nltk.pos_tag(text)

  def nltk_bigrams(self, text):
    return nltk.bigrams(text)

class tools:
  def parseXML(self, data):
    return parseString(data)

  def normalize(self, text):
    if not type(text) == unicode:
      text = unicode(text, errors='ignore')
      return text
    return unicodedata.normalize('NFKD', text).encode('ascii','ignore')

  def search_term_in_lines(self, term, lines):
    for i in range(len(lines)):
      l = lines[i]
      if term in l:
        return i
    # Cannot find, return the mid of the chunk
    return int(len(lines)/2)

  def check_if_number(self, term):
    return term.replace(',','').replace(',','').replace('%','').isdigit()

class weight:
  def __init__(self):
    self.sentence_tokenizer = PunktSentenceTokenizer()
    self.dist = dist()
    self.nltk_tools = nltk_tools()
    self.punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"
    self.stopwords = self.nltk_tools.nltk_stopwords()
    self.tools = tools()
    self.LAMBDA_AUTHOR_MATCH = 0.8
    reg = []
    reg.append(r"\(\s?(\d{1,3})\s?\)")
    reg.append(r"\(\s?(\d{4})\s?\)")
    reg.append(r"\(\s?(\d{4};?\s?)+\s?")
    reg.append(r"\[\s?(\d{1,3}\s?,?\s?)+\s?\]")
    reg.append(r"\[\s?([\w-],?\s?)+\s?\]")
    reg.append(r"et al\.?,?")
    # Complex regex
    reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?,?\s?(\(?(\d{4})\)?)")
    reg.append(r"([A-Z][A-Za-z-]+\s?,?\s?(\s(and|&)\s)?)+\s?,?\s?(et al\.?)?\s?,?\s?(\(?(\d{4})\)?)")
    self.regex = ""
    for i in range(len(reg)):
      self.regex += reg[i] + "|"
    self.regex = re.compile(self.regex[:-1])
    self.specific = ['obtains', 'obtained', 'score', 'scored', 'high',
              'F-score', 'Precision', 'precision', 'Recall', 'recall',
              'estimated', 'estimates', 'reported', 'reports',
              'probability', 'probabilities', 'peaked', 'experimental',
              'experimented', 'rate', 'error', 
              ]
    self.general = ['proposed', 'propose', 'presented', 'present', 'suggested'
            'suggests', 'described', 'describe', 'discuss', 'discussed',
            'gave', 'introduction', 'introduced', 'shown', 'showed',
            'sketched', 'sketch', 'talked', 'adopted', 'adopt', 'based',
            'originated', 'originate', 'built', 'researchers', 'comparative',
            'comparison', 'following', 'previously', 'previous'
              ]
    self.specific = map(lambda x: self.nltk_tools.nltk_stemmer(x), self.specific)
    self.specific = list(set(self.specific))
    self.general = map(lambda x: self.nltk_tools.nltk_stemmer(x), self.general)
    self.general = list(set(self.general))


  def chunk_average_weight(self, chunk, collection):
    temp_weight = 0
    if len(chunk.tokens) == 0:
      return 0
    for t in chunk.tokens:
      temp_weight += collection.tf_idf(t.lower(), chunk)
    return float(temp_weight) / float(len(chunk.tokens))
    
  def title_overlap(self, dom_parscit_section_citing, dom_parscit_section_cited):
    tags = ["title", "note", "booktitle", "journal", "tech", "author"]
    title_citing_tag = []
    for tag in tags:
      title_citing_tag = dom_parscit_section_citing.getElementsByTagName(tag)
      if title_citing_tag:
        break
    
    title_cited_tag = []
    for tag in tags:
      title_cited_tag = dom_parscit_section_cited.getElementsByTagName(tag)
      if title_cited_tag:
        break
    title_citing = title_citing_tag[0].firstChild.wholeText
    title_cited = title_cited_tag[0].firstChild.wholeText
    return self.dist.jaccard(title_citing, title_cited)

  def author_overlap(self, dom_parscit_section_citing, dom_parscit_section_cited):
    dom_authors_citing = dom_parscit_section_citing.getElementsByTagName('author')
    authors_citing = []
    for a in dom_authors_citing:
      authors_citing.append(a.firstChild.wholeText)
    authors_citing = list(Set(authors_citing))
    
    dom_authors_cited = dom_parscit_section_cited.getElementsByTagName('author')
    authors_cited = []
    for a in dom_authors_cited:
      authors_cited.append(a.firstChild.wholeText)
    authors_cited = list(Set(authors_cited))

    # Adapting the Jaccard idea
    matches = 0
    unique_names = len(authors_citing) + len(authors_cited)
    for citing_author in authors_citing:
      for cited_author in authors_cited:
        ratio = self.dist.levenshtein_ratio(citing_author, cited_author)
        if ratio > self.LAMBDA_AUTHOR_MATCH:
          matches += 1
          unique_names -= 1
    if unique_names == 0:
      return 1.0
    return float(matches) / float(unique_names)

  def physical_features(self, cit_str, context, dom_parscit_section_citing):
    cit_str = cit_str.replace("et al.", "et al")
    context = context.replace("et al.", "et al")
    context_lines = self.sentence_tokenizer.tokenize(context)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    before = context_lines[cit_sent-1] if (cit_sent-1 >= 0) else ""
    after = context_lines[cit_sent+1] if (cit_sent+1 < len(context_lines)) else ""
    cit_sent = context_lines[cit_sent]
    
    # Location
    location_vector = self.dist.cit_sent_location(cit_str, context, dom_parscit_section_citing)
    location = len(location_vector)-1
    for i in range(len(location_vector)):
      if location_vector[i] == 1:
        location = i

    # Popularity
    popularity = 0
    obj = re.findall(self.regex, cit_sent)
    popularity += len(obj)

    # Density
    uniq_references = []
    for l in [before, cit_sent, after]:
      obj = re.findall(self.regex, l)
      for o in obj:
        if o not in uniq_references:
          uniq_references.append(o)
    density = len(uniq_references)

    # AvgDens
    avg_dens = float(density) / float(len([before, cit_sent, after]))

    return (location, popularity, density, avg_dens)

  def cit_density(self, context_lines, cit_str):
    # Process citStr
    if "et al." in cit_str:
      cit_str = cit_str.replace("et al.", "et al")
    # Process context
    if "et al." in context_lines:
      context_lines = context_lines.replace("et al.", "et al")
    query_lines = self.sentence_tokenizer.tokenize(context_lines)
    citation_count = 0
    for l in query_lines:
      obj = re.findall(self.regex, l)
      citation_count += len(obj)
    avg_density = float(citation_count) / float(len(query_lines))
    return avg_density

  def number_density(self, cit_str, context):
    cit_str = cit_str.replace("et al.", "et al")
    context = context.replace("et al.", "et al")
    context_lines = self.sentence_tokenizer.tokenize(context)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    before = context_lines[cit_sent-1] if (cit_sent-1 >= 0) else ""
    after = context_lines[cit_sent+1] if (cit_sent+1 < len(context_lines)) else ""
    cit_sent = context_lines[cit_sent]
    this_regex = r"\d{2}\.\d+|(\d+,)+\d+|\d{2}\.?\d+%"

    # Popularity
    popularity = 0
    obj = re.findall(this_regex, cit_sent)
    popularity += len(obj)

    # Density
    uniq_numbers = []
    for l in [before, cit_sent, after]:
      obj = re.findall(this_regex, l)
      for o in obj:
        if o not in uniq_numbers:
          uniq_numbers.append(o)
    density = len(uniq_numbers)

    # AvgDens
    avg_dens = float(density) / float(len([before, cit_sent, after]))
    return (popularity, density, avg_dens)

  def pos_tag_distribution(self, cit_str, context):
    cit_str = cit_str.replace("et al.", "et al")
    context = context.replace("et al.", "et al")
    context_lines = self.sentence_tokenizer.tokenize(context)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    before = context_lines[cit_sent-1] if (cit_sent-1 >= 0) else ""
    after = context_lines[cit_sent+1] if (cit_sent+1 < len(context_lines)) else ""
    cit_sent = context_lines[cit_sent]

    cit_sent_text = self.nltk_tools.nltk_word_tokenize(cit_sent)
    cit_sent_pos = self.nltk_tools.nltk_pos(cit_sent_text)
    pos_hash = {}
    for (term, tag) in cit_sent_pos:
      if tag not in pos_hash:
        pos_hash[tag] = 0
      pos_hash[tag] += 1
    #print cit_sent
    #print cit_sent_pos
    #print sorted(pos_hash.iteritems(), key=operator.itemgetter(1))
    #print cit_str
    return pos_hash

  def cos_sim(self, query_tokens, query_col, doc_tokens, docs_col, vocab):
    # Vocab
    vocab = map(lambda x: x.lower(), vocab)
    vocab = [w for w in vocab if not w in self.stopwords]
    vocab = [w for w in vocab if not w in self.punctuation]

    # Prep Vectors
    u = []
    v = []
    temp_query = map(lambda x: x.lower(), query_tokens)
    temp_query = [w for w in temp_query if not w in self.stopwords]
    temp_query = [w for w in temp_query if not w in self.punctuation]
    temp_doc = map(lambda x: x.lower(), doc_tokens)
    temp_doc = [w for w in temp_doc if not w in self.stopwords]
    temp_doc = [w for w in temp_doc if not w in self.punctuation]
    for term in vocab:
      if term in temp_query:
        try:
          u.append(docs_col.tf_idf(term, temp_doc))
        except:
          u.append(0)
        #u.append(100)
      else:
        u.append(0)
        #u.append(1)
      if term in temp_doc:
        v.append(docs_col.tf_idf(term, temp_doc))
        #v.append(100)
      else:
        v.append(0)
        #v.append(1)
    if math.sqrt(np.dot(u, u)) == 0.0:
      return 0.0
    else:
      r = self.nltk_tools.nltk_cosine_distance(u,v)
      return r

  def cos_sim_bigrams(self, query_bigrams, doc_bigrams, bigrams_vocab):
    # Prep vectors
    u = []
    v = []
    for b in bigrams_vocab:
      if b in query_bigrams:
        u.append(100)
      else:
        u.append(1)
      if b in doc_bigrams:
        v.append(100)
      else:
        v.append(1)
    return self.nltk_tools.nltk_cosine_distance(u,v)

  def cosine_similarity(self, query_tokens, query_col, dom_parscit_section_cited):
    docs = []
    #body_texts = dom_parscit_section_cited.getElementsByTagName('bodyText')
    body_texts = dom_parscit_section_cited.getElementsByTagName('variant')[0].childNodes
    for body_text in body_texts:
      if body_text.nodeType == body_text.TEXT_NODE:
        continue
      whole_text = body_text.firstChild.wholeText.lower()
      whole_text = unicode(whole_text.encode('ascii', 'ignore'), errors='ignore')
      text = self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(whole_text.lower()))
      docs.append(text)
    docs_col = self.nltk_tools.nltk_text_collection(docs)
    
    # Vocab
    vocab = list(set(query_col) | set(docs_col))
    vocab = map(lambda x: x.lower(), vocab)
    vocab = [w for w in vocab if not w in self.stopwords]
    vocab = [w for w in vocab if not w in self.punctuation]

    # Prep Vectors
    results = []
    for i in range(0, len(docs)):
      # Cited Chunk's Average TF-IDF Weight
      chunk_avg_weight = self.chunk_average_weight(docs[i], docs_col)

      u = []
      v = []
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
        r = self.nltk_tools.nltk_cosine_distance(u,v)
        results.append((chunk_avg_weight,r))
    feature = []
    for i in range(len(results)):
      feature.append((docs[i],results[i]))
    return feature

  def cue_words(self, cit_str, context):
    cit_str = cit_str.replace("et al.", "et al")
    context = context.replace("et al.", "et al")
    context_lines = self.sentence_tokenizer.tokenize(context)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    before = context_lines[cit_sent-1] if (cit_sent-1 >= 0) else ""
    after = context_lines[cit_sent+1] if (cit_sent+1 < len(context_lines)) else ""
    cit_sent = context_lines[cit_sent]

    # Popularity
    popularity_specific = 0
    cit_sent_tokens = self.nltk_tools.nltk_word_tokenize(cit_sent)
    cit_sent_stemmed = map(lambda x: self.nltk_tools.nltk_stemmer(x), cit_sent_tokens)
    for c in self.specific:
      if c in cit_sent_stemmed:
        popularity_specific += 1
    popularity_general = 0
    cit_sent_tokens = self.nltk_tools.nltk_word_tokenize(cit_sent)
    cit_sent_stemmed = map(lambda x: self.nltk_tools.nltk_stemmer(x), cit_sent_tokens)
    for c in self.general:
      if c in cit_sent_stemmed:
        popularity_general += 1

    # Density
    uniq_specific = []
    uniq_general = []
    for l in [before, cit_sent, after]:
      l_tokens = self.nltk_tools.nltk_word_tokenize(l)
      l_stemmed = map(lambda x: self.nltk_tools.nltk_stemmer(x), l_tokens)
      for c in self.specific:
        if c in l_stemmed:
          if c not in uniq_specific:
            uniq_specific.append(c)
      for c in self.general:
        if c in l_stemmed:
          if c not in uniq_general:
            uniq_general.append(c)
    density_specific = len(uniq_specific)
    density_general = len(uniq_general)
    
    # AvgDens
    avg_dens_specific = float(density_specific) / float(len([before, cit_sent, after]))
    avg_dens_general = float(density_general) / float(len([before, cit_sent, after]))
    
    return (popularity_specific, density_specific, avg_dens_specific, popularity_general, density_general, avg_dens_general)

  def surface_matching(self, cit_sent_tokens, cit_sent_text_tagged, candidate_text):
    temp_query = map(lambda x: x.lower(), cit_sent_tokens)
    temp_query = [w for w in temp_query if not w in self.stopwords]
    temp_query = [w for w in temp_query if not w in self.punctuation]
    temp_candidate = map(lambda x: x.lower(), candidate_text)
    temp_candidate = [w for w in temp_candidate if not w in self.stopwords]
    temp_candidate = [w for w in temp_candidate if not w in self.punctuation]

    num_check = []
    text_check = []
    for (term, tag) in cit_sent_text_tagged:
      if tag == 'CD':
        num_check.append(term)
      else:
        text_check.append(term)

    count = float(0)
    for term in num_check:
      if term in temp_candidate:
        count += 1
    num_match = count / float(len(temp_query))
    count = float(0)
    for term in text_check:
      if term in temp_candidate:
        count += 1
    text_match = count / float(len(temp_query))
    return (num_match, text_match)

  def number_near_miss(self, cit_sent_tokens, cit_sent_text_tagged, candidate_text):
    candidate_tagged = self.nltk_tools.nltk_pos(candidate_text)
    # Filter to look at only numbers like 81.1, 81.06, 0.92, 92
    cit_sent_num_only = []
    candidate_num_only = []
    for (term, tag) in cit_sent_text_tagged:
      if tag == 'CD':
        try:
          temp = float(term)
          if temp < 100:
            cit_sent_num_only.append(temp)
        except:
          None
    for (term, tag) in candidate_tagged:
      if tag == 'CD':
        try:
          temp = float(term)
          if temp < 100:
            candidate_num_only.append(temp)
        except:
          None
    if len(cit_sent_num_only) == 0 or len(candidate_num_only) == 0:
      return 0.0

    candidate_num_only_cleaned = []
    for term in candidate_num_only:
      if term > 1:
        # Rounding
        num_dp = abs(decimal.Decimal(term).as_tuple().exponent)
        candidate_num_only_cleaned.append(round(term,1))
      elif term < 1:
        # Percentage
        candidate_num_only_cleaned.append(term*100)
    candidate_num_only_cleaned = list(set(candidate_num_only_cleaned))
    count = float(0)
    for term in candidate_num_only_cleaned:
      if term in cit_sent_num_only:
        count += 1
    return count / float(len(cit_sent_num_only))

  def bigrams_matching(self, citing_bigrams, doc_bigrams):
    count = float(0)
    for tup in citing_bigrams:
      if tup in doc_bigrams:
        count += 1
    value = count / float(len(citing_bigrams))
    return value
  
  def referToDefinition(self, cit_str, context):
    print

  def referToQuote(self, cit_str, context):
    print

  def referToMethod(self, cit_str, context):
    print

class dist:
  def __init__(self):
    self.sentence_tokenizer = PunktSentenceTokenizer()
    self.tools = tools()
    self.generic_header = ['abstract',
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

  def levenshtein_ratio(self, a, b):
    lensum = float(len(a) + len(b))
    if lensum == 0.0:
      return 1.0
    ldist = self.levenshtein(a,b)
    return (lensum - ldist) / lensum

  def jaccard(self, inputA, inputB):
    if type(inputA) == str or type(inputA) == unicode:
      inputA_tokens = inputA.lower().split()
      inputB_tokens = inputB.lower().split()
      return distance.jaccard_distance(set(inputA_tokens), set(inputB_tokens))
    elif type(inputA) == nltk.text.Text:
      inputA_tokens = map(lambda x: x.lower(), inputA.vocab().keys())
      inputB_tokens = map(lambda x: x.lower(), inputB.vocab().keys())
      return distance.jaccard_distance(set(inputA_tokens), set(inputB_tokens))
    elif type(inputA) == list:
      inputA_tokens = map(lambda x: x.lower(), inputA)
      inputB_tokens = map(lambda x: x.lower(), inputB)
      return distance.jaccard_distance(set(inputA_tokens), set(inputB_tokens))

  def masi(self, a, b):
    a = a.lower()
    b = b.lower()
    return distance.masi_distance(set(a.split()), set(b.split()))

  def publish_year(self, cite_key):
    citing = cite_key['citing']
    cited = cite_key['cited']

    citing_year = int(citing[1:3])
    cited_year = int(cited[1:3])

    if citing_year > 50:
      citing_year = 1900 + citing_year
    else:
      citing_year = 2000 + citing_year

    if cited_year > 50:
      cited_year = 1900 + cited_year
    else:
      cited_year = 2000 + cited_year
    return (citing_year-cited_year)

  def cit_sent_location(self, cit_str, query, dom_parscit_section_citing):
    vector = []
    cit_str = cit_str.replace("et al.", "et al")
    query = query.replace("et al.", "et al")

    context_lines = self.sentence_tokenizer.tokenize(query)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    target = None
    body_texts = dom_parscit_section_citing.getElementsByTagName('bodyText')
    this_regex = r"\<.*\>(.*)\<.*\>"

    min_distance = 314159265358979323846264338327950288419716939937510
    for i in range(len(body_texts)):
      b = body_texts[i]
      #text = tool.normalize(b.toxml().replace("\n", " ").replace("- ", "").strip())
      text = b.toxml().replace("\n", " ").replace("- ", "").strip()
      obj = re.findall(this_regex, text)
      temp_dist = self.jaccard(context_lines[cit_sent], text)
      if temp_dist < min_distance:
        min_distance = temp_dist
        target = b
    
    if target:
      searching = True
      section_header_node = None
      target = target.previousSibling
      while target:
        if target.nodeType == Node.ELEMENT_NODE:
          if target.nodeName == 'sectionHeader':
            section_header_node = target
            break
        target = target.previousSibling
      if target == None:
        for h in (self.generic_header):
          vector.append(0)
        vector[-1] = 1 # Setting 'None' to 1
        return vector
      if section_header_node.attributes.has_key('genericHeader'):
        header = section_header_node.attributes['genericHeader'].value
      elif section_header_node.attributes.has_key('genericheader'):
        header = section_header_node.attributes['genericheader'].value
      for h in (self.generic_header):
        if header == h:
          vector.append(1)
        else:
          vector.append(0)
      return vector
    else:
      # Not found
      for h in (self.generic_header):
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
  def __init__(self, dist, nltk_tools, pickler, tools):
    self.dist = dist
    self.nltk_tools = nltk_tools
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
    title_tag = []
    index = 0
    best_index = -1
    min_distance = 314159265358979323846264338327950288419716939937510
    for i in range(len(citations)):
      c = citations[i]
      #valid = c.getAttribute('valid')
      #if valid == "true":
      title_tag = []
      for index in range(len(tags)):
        title_tag = c.getElementsByTagName(tags[index])
        if title_tag:
          break
      if title_tag == [] or titleTag[0].firstChild == None:
        continue
      title = title_tag[0].firstChild.data
      if not type(title) == unicode:
        title = tools.normalize(title)
      if re.search("Computational Linguistics,$", title):
        title = title.replace("Computational Linguistics,", "")
      levenshtein_distance = dist.levenshtein(title.lower(), titleToMatch.lower())
      masi_distance = dist.masi(title, titleToMatch)
      this_distance = levenshtein_distance*masi_distance
      if this_distance < min_distance:
        min_distance = this_distance
        best_index = i
    if best_index == -1:
      return None
    return citations[best_index]

  def prepContextsCFS(self, dist, tools, title_citing, title_cited, dom_citing_parscit):
    titleToMatch = title_cited
    dom = dom_citing_parscit

    citations = dom.getElementsByTagName('citation')
    tags = ["title", "note", "booktitle", "journal", "tech", "author"]
    title_tag = []
    index = 0
    best_index = -1
    min_distance = 314159265358979323846264338327950288419716939937510
    for i in range(len(citations)):
      c = citations[i]
      title_tag = []
      for index in range(len(tags)):
        title_tag = c.getElementsByTagName(tags[index])
        if title_tag:
          break
      if title_tag == [] or titleTag[0].firstChild == None:
        continue
      title = title_tag[0].firstChild.data
      if not type(title) == unicode:
        title = tools.normalize(title)
      if re.search("Computational Linguistics,$", title):
        title = title.replace("Computational Linguistics,", "")
      levenshtein_distance = dist.levenshtein(title.lower(), titleToMatch.lower())
      masi_distance = dist.masi(title, titleToMatch)
      this_distance = levenshtein_distance*masi_distance
      if this_distance < min_distance:
        min_distance = this_distance
        best_index = i
    if best_index == -1:
      return None
    return citations[best_index]

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
        context_list.append(self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(value)))
      citing_col = self.nltk_tools.nltk_text_collection(context_list)
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
          # i[1][1] is chunk_avg_weight
          temp.append(i[1][1])
          # i[1][0] is cosine_sim
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
        context_list.append(self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(value)))
      citing_col = self.nltk_tools.nltk_text_collection(context_list)
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
        context_list.append(self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(value)))
      citing_col = self.nltk_tools.nltk_text_collection(context_list)
      for c in contexts:
        x = run.extractFeaturesCFS(e, c, citing_col)
        forannotation.append((e, c))
        keys.append(e)
        dataset.append(x)
    X = np.asarray(dataset)
    return (forannotation, keys, X)

  def prepAnnotations(self, annotationFile):
    this_regex = r"\#(\d{3})\s+(.*)==>(.*),(.*)"
    target = []
    for l in open(annotationFile):
      l = l.strip()
      obj = re.findall(this_regex, l)
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

class extract_features:
  def __init__(self):
    self.dist = dist()
    self.nltk_tools = nltk_tools()
    self.tools = tools()
    self.weight = weight()
    self.stopwords = self.nltk_tools.nltk_stopwords()
    self.punctuation = "~`!@#$%^&*()-_+={}[]|\\:;\"\'<>,.?/"

  def extract_feature_1st_tier(self, f, context, citing_col, dom_parscit_section_citing, dom_parscit_section_cited):
    cit_str = context.getAttribute('citStr')
    cit_str = unicode(cit_str.encode('ascii','ignore'), errors='ignore')
    query = context.firstChild.wholeText
    query = unicode(query.encode('ascii','ignore'), errors='ignore')

    query_tokens = self.nltk_tools.nltk_word_tokenize(query.lower())
    query_text = self.nltk_tools.nltk_text(query_tokens)
    query_col = self.nltk_tools.nltk_text_collection([query_text])

    # Extract Features
    x = []
    
    # Citation Density
    #feature_cit_density = self.weight.cit_density(query, cit_str)
    #x.append(feature_cit_density)

    # Physical Features
    feature_physical = self.weight.physical_features(cit_str, query, dom_parscit_section_citing)
    for i in feature_physical:
      x.append(i)

    # Number Density
    feature_num_density = self.weight.number_density(cit_str, query)
    for i in feature_num_density:
      x.append(i)

    # Publishing Year Difference
    feature_publish_year = self.dist.publish_year(f)
    x.append(feature_publish_year)

    # Title Overlap
    #feature_title_overlap = self.weight.title_overlap(dom_parscit_section_citing, dom_parscit_section_cited)
    #x.append(feature_title_overlap)

    # Authors Overlap
    #feature_author_overlap = self.weight.author_overlap(dom_parscit_section_citing, dom_parscit_section_cited)
    #x.append(feature_author_overlap)

    # Context's Average TF-IDF Weight
    feature_query_weight = self.weight.chunk_average_weight(query_text, citing_col)
    x.append(feature_query_weight)

    # Location of Citing Sentence
    #feature_cit_sent_location = self.dist.cit_sent_location(cit_str, query, dom_parscit_section_citing)
    #x.extend(feature_cit_sent_location)

    # Cue Words
    feature_cue_words = self.weight.cue_words(cit_str, query)
    for i in feature_cue_words:
      x.append(i)

    # Trying out POS tagging
    #feature_pos_tag = self.weight.pos_tag_distribution(cit_str, query)

    return x
  
  def extract_feature_1st_tier_baseline(self, f, context, citing_col, dom_parscit_section_citing, dom_parscit_section_cited):
    cit_str = context.getAttribute('citStr')
    cit_str = unicode(cit_str.encode('ascii','ignore'), errors='ignore')
    query = context.firstChild.wholeText
    query = unicode(query.encode('ascii','ignore'), errors='ignore')

    query_tokens = self.nltk_tools.nltk_word_tokenize(query.lower())
    query_text = self.nltk_tools.nltk_text(query_tokens)
    query_col = self.nltk_tools.nltk_text_collection([query_text])

    # Use Cosine Similarity as Baseline
    big_doc = ""
    docs = []
    vocab = []
    body_texts = dom_parscit_section_cited.getElementsByTagName('variant')[0].childNodes
    for body_text in body_texts:
      if body_text.nodeType == body_text.TEXT_NODE:
        continue
      whole_text = body_text.firstChild.wholeText.lower()
      whole_text = unicode(whole_text.encode('ascii', 'ignore'), errors='ignore')
      whole_text = whole_text.replace(cit_str, "")
      big_doc += whole_text + " "
      text = self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(whole_text.lower()))
      docs.append(text)
      vocab.extend(whole_text.split())
    docs_col = self.nltk_tools.nltk_text_collection(docs)
    vocab.extend(query_tokens)
    big_doc_tokens = self.nltk_tools.nltk_word_tokenize(big_doc.lower())

    feature_cos_sim = self.weight.cos_sim(query_tokens, citing_col, big_doc_tokens, docs_col, vocab)
    return feature_cos_sim
  
  def extract_feature_2nd_tier(self, f, c, citing_col, dom_parscit_section_citing, dom_parscit_section_cited):
    # Fix encoding problems
    cit_str = c.getAttribute('citStr')
    cit_str = unicode(cit_str.encode('ascii','ignore'), errors='ignore')
    context = c.firstChild.wholeText
    context = unicode(context.encode('ascii','ignore'), errors='ignore')
    
    # Getting the citing sentence and its neighbour sentences
    cit_str = cit_str.replace("et al.", "et al")
    context = context.replace("et al.", "et al")
    context_lines = self.dist.sentence_tokenizer.tokenize(context)
    cit_sent = self.tools.search_term_in_lines(cit_str, context_lines)
    before = context_lines[cit_sent-1] if (cit_sent-1 >= 0) else ""
    after = context_lines[cit_sent+1] if (cit_sent+1 < len(context_lines)) else ""
    cit_sent = context_lines[cit_sent]
    cit_sent_no_cit_str = cit_sent.replace(cit_str, "")

    before_tokens = self.nltk_tools.nltk_word_tokenize(before.lower())
    before_text = self.nltk_tools.nltk_text(before_tokens)
    before_bigrams = self.nltk_tools.nltk_bigrams(before_text)
    
    cit_sent_tokens = self.nltk_tools.nltk_word_tokenize(cit_sent.lower())
    cit_sent_text = self.nltk_tools.nltk_text(cit_sent_tokens)
    cit_sent_bigrams = self.nltk_tools.nltk_bigrams(cit_sent_text)
    cit_sent_text_tagged = self.nltk_tools.nltk_pos(cit_sent_text)
    cit_sent_no_cit_str_tokens = self.nltk_tools.nltk_word_tokenize(cit_sent_no_cit_str.lower())
    cit_sent_no_cit_str_text = self.nltk_tools.nltk_text(cit_sent_no_cit_str_tokens)
    cit_sent_no_cit_str_bigrams = self.nltk_tools.nltk_bigrams(cit_sent_no_cit_str_text)
    cit_sent_no_cit_str_text_tagged = self.nltk_tools.nltk_pos(cit_sent_no_cit_str_text)
    
    after_tokens = self.nltk_tools.nltk_word_tokenize(after.lower())
    after_text = self.nltk_tools.nltk_text(after_tokens)
    after_bigrams = self.nltk_tools.nltk_bigrams(after_text)

    context_tokens = self.nltk_tools.nltk_word_tokenize(context.lower())
    context_text = self.nltk_tools.nltk_text(context_tokens)
    context_bigrams = self.nltk_tools.nltk_bigrams(context_text)
    context_col = self.nltk_tools.nltk_text_collection([context_text])

    # Setting up vocab, bigrams_vocab
    docs = []
    bigrams_vocab = []
    vocab = []
    body_texts = dom_parscit_section_cited.getElementsByTagName('variant')[0].childNodes
    for body_text in body_texts:
      if body_text.nodeType == body_text.TEXT_NODE:
        continue
      whole_text = body_text.firstChild.wholeText.lower()
      whole_text = unicode(whole_text.encode('ascii', 'ignore'), errors='ignore')
      whole_text = whole_text.replace(cit_str, "")
      text = self.nltk_tools.nltk_text(self.nltk_tools.nltk_word_tokenize(whole_text.lower()))
      bigrams_vocab.extend(self.nltk_tools.nltk_bigrams(text))
      docs.append(text)
      vocab.extend(whole_text.split())
    docs_col = self.nltk_tools.nltk_text_collection(docs)
    bigrams_vocab = set(bigrams_vocab)
    vocab.extend(context_tokens)
    vocab = set(vocab)

    # Extract Features
    # In 2nd tier, features are to match context to specific chunk
    # For each chunk, we extract a feature vector, hence we return a list of feature vectors
    X = []
    x = []
    max_sim = -1
    max_index = 0
    for i in range(len(docs)):
      doc = docs[i]
      # Extract features for each body_text
      x = []

      # Surface Matching - Using only cit_sent
      # returns (num_match, text_match)
      feature_surface_matching = self.weight.surface_matching(cit_sent_tokens, cit_sent_text_tagged, doc)
      for feat in feature_surface_matching:
        x.append(feat)

      # Number Near Miss e.g. 81.1 for 81.06, 92 for 0.92
      feature_number_near_miss = self.weight.number_near_miss(cit_sent_tokens, cit_sent_text_tagged, doc)
      x.append(feature_number_near_miss)

      # Bigrams Matching
      doc_tokens = []
      for k in doc:
        doc_tokens.append(k)
      doc_tokens = [w for w in doc_tokens if not w in self.stopwords]
      doc_tokens = [w for w in doc_tokens if not w in self.punctuation]
      doc_bigrams = self.nltk_tools.nltk_bigrams(doc_tokens)
      doc_bigrams = list(set(doc_bigrams))
      feature_bigrams_matching = self.weight.bigrams_matching(cit_sent_no_cit_str_bigrams, doc_bigrams)
      x.append(feature_bigrams_matching)

      # Cos sim
      feature_cos_sim = self.weight.cos_sim(cit_sent_tokens, context_col, doc, docs_col, vocab)
      x.append(feature_cos_sim)
      
      X.append(x)
    #print "############"
    #print cit_sent_tokens
    #print max_index
    #print max_sim
    #print docs[max_index].vocab().keys()
    return X

  def extract_feature(self, f, context, citing_col, dom_parscit_section_citing, dom_parscit_section_cited):
    cit_str = context.getAttribute('citStr')
    cit_str = unicode(cit_str.encode('ascii','ignore'), errors='ignore')
    query = context.firstChild.wholeText
    query = unicode(query.encode('ascii','ignore'), errors='ignore')

    query_tokens = self.nltk_tools.nltk_word_tokenize(query.lower())
    query_text = self.nltk_tools.nltk_text(query_tokens)
    query_col = self.nltk_tools.nltk_text_collection([query_text])

    # Extract Features
    X = []
    x = []
    
    # Citation Density
    feature_cit_density = self.weight.cit_density(query, cit_str)
    x.append(feature_cit_density)

    # Publishing Year Difference
    feature_publish_year = self.dist.publish_year(f)
    x.append(feature_publish_year)

    # Title Overlap
    feature_title_overlap = self.weight.title_overlap(dom_parscit_section_citing, dom_parscit_section_cited)
    x.append(feature_title_overlap)

    # Authors Overlap
    feature_author_overlap = self.weight.author_overlap(dom_parscit_section_citing, dom_parscit_section_cited)
    x.append(feature_author_overlap)

    # Context's Average TF-IDF Weight
    feature_query_weight = self.weight.chunk_average_weight(query_text, citing_col)
    x.append(feature_query_weight)

    # Location of Citing Sentence
    feature_cit_sent_location = self.dist.cit_sent_location(cit_str, query, dom_parscit_section_citing)
    x.extend(feature_cit_sent_location)

    # Cosine Similarity
    feature_cosine_similarity = self.weight.cosine_similarity(query_tokens, query_col, dom_parscit_section_cited)
    for i in feature_cosine_similarity:
      cosine_tuple = i[1]
      chunk_avg_weight = cosine_tuple[0]
      cosine_sim = cosine_tuple[1]
      temp = x[:]
      temp.append(chunk_avg_weight)
      temp.append(cosine_sim)
      X.append(temp)
    return X
