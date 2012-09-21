import cPickle as pickle
import Utils
import os
import numpy as np

if __name__ == "__main__":
  # Output
  big_X = []
  extract_features = Utils.extract_features()
  nltk_tools = Utils.nltk_tools()

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  # Load Filtered pickle
  filtered = pickle.load(open(os.path.join(path_pickles,'Filtered.pickle'),'r'))
  # Load Contexts pickle
  contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
  # Load Doms pickle
  doms = pickle.load(open(os.path.join(path_pickles,'Doms.pickle'),'r'))
  num = len(filtered)
  for f in filtered:
    citing = f['citing']
    cited = f['cited']
    hash_key = citing + "==>" + cited

    f_contexts = contexts[hash_key]
    f_dom = doms[hash_key]
    dom_parscit_citing = f_dom[0]
    dom_parscit_section_citing = f_dom[1]
    dom_parscit_cited = f_dom[2]
    dom_parscit_section_cited = f_dom[3]
    context_list = []
    for c in f_contexts:
      value = c.firstChild.wholeText
      value = unicode(value.encode('ascii', 'ignore'), errors='ignore')
      context_list.append(nltk_tools.nltk_text(nltk_tools.nltk_word_tokenize(value)))
    citing_col = nltk_tools.nltk_text_collection(context_list)
    for c in f_contexts:
      X = extract_features.extract_feature(c, citing_col, dom_parscit_section_citing, dom_parscit_section_cited)
      big_X.extend(X)
    num -= 1
    print "No. left = " + str(num)
  big_X = np.asarray(big_X)
  
  # Dump pickle
  pickle.dump(open(os.path.join(path_pickles,'Big_X.pickle'),'wb'))
