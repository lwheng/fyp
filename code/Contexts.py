import cPickle as pickle
import os
import re
import Utils

if __name__ == '__main__':
  # Input
  dist = Utils.dist()
  # Output
  contexts = {}

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Filtered
  filtered = pickle.load(open(os.path.join(path_pickles, "Filtered.pickle"),'r'))
  # Load Doms
  doms = pickle.load(open(os.path.join(path_pickles, "Doms.pickle"),'r'))

  for f in filtered:
    citing = f['citing']
    cited = f['cited']
    hash_key = citing + "==>" + cited
    d = doms[hash_key]

    # Take parscit_citing and parscit_section_cited
    dom_parscit_citing = d[0]
    dom_parscit_section_cited = d[3]

    title_to_match = dom_parscit_section_cited.getElementsByTagName('title')[0].firstChild.wholeText

    # Citations in citing
    citations = dom_parscit_citing.getElementsByTagName('citation')
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
      if title_tag == [] or title_tag[0].firstChild == None:
        continue
      title = title_tag[0].firstChild.wholeText
      #if not type(title) == unicode:
      #  title = tools.normalize(title)
      if re.search("Computational Linguistics,$", title):
        title = title.replace("Computational Linguistics,", "")
      levenshtein_distance = dist.levenshtein(title.lower(), title_to_match.lower())
      masi_distance = dist.masi(title, title_to_match)
      thisDistance = levenshtein_distance*masi_distance
      if thisDistance < min_distance:
        min_distance = thisDistance
        best_index = i
    if best_index == -1:
      citation = None
    else:
      citation = citations[best_index]
    this_contexts = citation.getElementsByTagName('context')
    contexts[hash_key] = this_contexts

  # Dump pickle
  pickle.dump(contexts, open(os.path.join(path_pickles, "Contexts.pickle")))
