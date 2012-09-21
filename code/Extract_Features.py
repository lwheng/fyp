import cPickle as pickle
import Utils
import os
import numpy as np

if __name__ == "__main__":
  # Output
  big_X = []
  extract_features = Utils.extract_features()

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
    for c in f_contexts:
      try:
        X = extract_features.extract_feature(c, dom_parscit_section_citing, dom_parscit_section_cited)
      except:
        print f
      big_X.extend(X)
    num -= 1
    print "No. left = " + str(num)
  big_X = np.asarray(big_X)
  
  # Dump pickle
  pickle.dump(open(os.path.join(path_pickles,'Big_X.pickle'),'wb'))
