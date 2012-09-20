import cPickle as pickle
import Utils
import os

class Extract_Features:
  def __init__(self):
    print

  def extract_feature(self, context, dom_parscit_section_citing, dom_parscit_section_cited)
    cit_str = context.getAttribute('citStr')
    cit_str = self.tools.normalize(cit_str)
    query = context.firstChild.wholeText
    query = self.tools.normalize(query)

    query_tokens = self.nltk_Tools.nltkWordTokenize(query.lower())
    query_text = self.nltk_Tools.nltkText(query_tokens)
    query_col = self.nltk_Tools.nltkTextCollection([query_text])

    # Extract Features
    X = []
    x = []
    
    # Citation Density
    feature_citDensity = self.weight.citDensity(query, citStr)
    x.append(feature_citDensity)

    # Publishing Year Difference
    feature_publishYear = self.dist.publishYear(dom_parscit_section_citing, dom_parscit_section_cited)
    x.append(feature_publishYear)

    # Title Overlap
    feature_titleOverlap = self.weight.titleOverlap(dom_parscit_section_citing, dom_parscit_section_cited)
    x.append(feature_titleOverlap)

    # Authors Overlap
    feature_authorOverlap = self.weight.authorOverlap(dom_parscit_section_citing, dom_parscit_section_cited)
    x.append(feature_authorOverlap)

    # Context's Average TF-IDF Weight
    feature_queryWeight = self.weight.chunkAverageWeight(query_text, citing_col)
    x.append(feature_queryWeight)

    # Location of Citing Sentence
    feature_locationCitSent = self.dist.citSentLocation(citStr, query, dom_parscit_section_citing)
    x.extend(feature_locationCitSent)

    # Cosine Similarity
    feature_cosineSimilarity = self.cosineSimilarity(cite_key, query_tokens, query_col, self.pickler.pathPDFBox)
    x.append(feature_cosineSimilarity)
    return x
