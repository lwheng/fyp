execfile("Utils.py")
import Utils
import cPickle as pickle
import os
import sys
import numpy as np
from sklearn import svm
from sklearn import cross_validation

if __name__ == "__main__":
  labels_to_index = {'g':0, 'y':1, 'n':1}
  nltk_tools = Utils.nltk_tools()
  printer = Utils.printer()
  string = "Building 1st Tier Model"
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
  # Load y_hash_1st_tier
  y_hash_1st_tier = pickle.load(open(os.path.join(path_pickles, 'Y_Hash_1st_Tier.pickle'),'r'))
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
  for f in filtered:
    citing = f['citing']
    cited = f['cited']
    hash_key = citing+"==>"+cited
    if hash_key in y_hash_1st_tier:
      f_contexts = contexts[hash_key]
      context_list = []
      for c in f_contexts:
        value = c.firstChild.wholeText
        value = unicode(value.encode('ascii', 'ignore'), errors='ignore')
        context_list.append(nltk_tools.nltk_text(nltk_tools.nltk_word_tokenize(value)))
      citing_col = nltk_tools.nltk_text_collection(context_list)
      for i in range(len(f_contexts)):
        if y_hash_1st_tier[hash_key][i] == 'u':
          continue
        c = f_contexts[i]
        x = feature_extractor.extract_feature_1st_tier(f, c, citing_col, doms[hash_key][1], doms[hash_key][3])
        X.append(x)
        y.append(labels_to_index[y_hash_1st_tier[hash_key][i]])
  X = np.asarray(X)
  y = np.asarray(y)

  print X
  print "X = " + str(X.shape)
  print y
  print "y = " + str(y.shape)

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

  ## Write out the Model
  pickle.dump(model, open(os.path.join(path_pickles, 'Model_1st_Tier.pickle'),'wb'))

  # Prediction
  X_g = []
  y_g = []
  X_s = []
  y_s = []
  for i in range(y.shape[0]):
    temp_x = list(X[i])
    temp_y = int(y[i])
    if temp_y == 1:
      X_s.append(temp_x)
      y_s.append(temp_y)
    else:
      X_g.append(temp_x)
      y_g.append(temp_y)
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  X_train.append(X_g[0])
  y_train.append(y_g[0])
  X_train.append(X_s[0])
  y_train.append(y_s[0])
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)
  print X_train
  print y_train
  clf = svm.SVC(kernel='linear')
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_train)
  print predicted
