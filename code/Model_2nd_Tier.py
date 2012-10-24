execfile("Utils.py")
import Utils
import cPickle as pickle
import os
import sys
import numpy as np
import random
from sklearn import svm

if __name__ == "__main__":
  labels_to_index = {'g':0, 'y':1, 'n':1}
  nltk_tools = Utils.nltk_tools()
  printer = Utils.printer()
  string = "Building 2nd Tier Model"
  print printer.line_printer(len(string), "#")
  print string
  print printer.line_printer(len(string), "#")

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  ## Input To Model
  # Annotations
  string = "Loading Annotations"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")
  # Run Targets.py to generate Y.pickle from Labels.txt
  execfile("Targets.py")
  # Load y_hash_2nd_tier
  y_hash_2nd_tier = pickle.load(open(os.path.join(path_pickles, 'Y_Hash_2nd_Tier.pickle'),'r'))
  string = "Loaded Annotations"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")

  # Doms
  string = "Loading Doms"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")
  doms = pickle.load(open(os.path.join(path_pickles, "Doms.pickle"),'r'))
  string = "Loaded Doms"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")

  # Contexts
  string = "Loading Contexts"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")
  contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
  filtered = pickle.load(open(os.path.join(path_pickles, "Filtered.pickle"),'r'))
  string = "Loaded Contexts. Extract Features"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")
  feature_extractor = Utils.extract_features()
  X = []
  y = []
  # Extract Features
  # We should not use filtered this time. Use keys from y_hash_2nd_tier instead
  second_tier_keys = y_hash_2nd_tier.keys()
  num = len(second_tier_keys)
  for f in second_tier_keys:
    f = "N04-1032==>C02-1126" # REMOVE AFTER DEBUG
    info = f.split("==>")
    citing = info[0]
    cited = info[1]
    hash_key = citing+"==>"+cited
    f_contexts = contexts[hash_key]
    f_hash = y_hash_2nd_tier[f]
    dom_parscit_section_cited = doms[hash_key][3]
    f_contexts = contexts[hash_key]
    context_list = []
    for context_id in f_hash.keys():
      c = f_contexts[context_id]
      value = c.firstChild.wholeText
      value = unicode(value.encode('ascii', 'ignore'), errors='ignore')
      context_list.append(nltk_tools.nltk_text(nltk_tools.nltk_word_tokenize(value)))
    citing_col = nltk_tools.nltk_text_collection(context_list)
    for context_id in f_hash.keys():
      c = f_contexts[context_id]
      c_hash = f_hash[context_id]
      x = feature_extractor.extract_feature_2nd_tier(f, c, citing_col, doms[hash_key][1], doms[hash_key][3])
      # x would be a list of feature vector because we are comparing 1 context against bodyTexts
      X.extend(x)
      for i in range(len(x)):
        if i in c_hash:
          # Append y
          y.append(1)
        else:
          # Append n
          y.append(0)
    sys.exit()
    num -= 1
    print "No. of keys left = " + str(num)
  X = np.asarray(X)
  y = np.asarray(y)

  ## Fit the Model
  string = "Fitting Model"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")
  # Select Classifier
  model = svm.SVC(kernel='linear')
  model.fit(X[0:len(y)], y)
  # Fit X and y
  string = "Fitted Model. Writing Out Now"
  print printer.line_printer(len(string), "-")
  print string
  print printer.line_printer(len(string), "-")

  ## Write out X
  pickle.dump(X, open(os.path.join(path_pickles, 'X_2nd_Tier.pickle'),'wb'))
  ## Write out y
  pickle.dump(y, open(os.path.join(path_pickles, 'y_2nd_Tier.pickle'),'wb'))
  ## Write out the Model
  pickle.dump(model, open(os.path.join(path_pickles, 'Model_2nd_Tier.pickle'),'wb'))

  sys.exit()

  # Prediction
  X_n = []
  y_n = []
  X_y = []
  y_y = []
  for i in range(y.shape[0]):
    temp_x = list(X[i])
    temp_y = int(y[i])
    if temp_y == 1:
      X_y.append(temp_x)
      y_y.append(temp_y)
    else:
      X_n.append(temp_x)
      y_n.append(temp_y)
  X_train = X_y
  y_train = y_y
  
  # Pick randomly
  times = 2
  X_train.extend(random.sample(X_g, int(len(y_train)*times)))
  y_train.extend(random.sample(y_g, int(len(y_train)*times)))
