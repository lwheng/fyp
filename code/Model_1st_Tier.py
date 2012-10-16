execfile("Utils.py")
import Utils
import cPickle as pickle
import os

if __name__ == "__main__":
  print "#######################"
  print "Building 1st Tier Model"
  print "#######################"

  # Load Config.pickle
  config = pickle.load(open('Config.pickle','r'))
  path_parscit = config['path_parscit']
  path_parscit_section = config['path_parscit_section']
  path_pickles = config['path_pickles']

  ## Input To Model
  # Contexts
  print "----------------"
  print "Loading Contexts"
  print "----------------"
  contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
  print "---------------------------------"
  print "Loaded Contexts. Extract Features"
  print "---------------------------------"
  feature_extractor = Utils.extract_features()

  # Annotations
  print "-------------------"
  print "Loading Annotations"
  print "-------------------"
  # Run Targets.py to generate Y.pickle from Labels.txt
  execfile("Targets.py")
  # Load y_raw
  y_raw = pickle.load(open(os.path.join(path_pickles, 'Y.pickle'),'r'))
  y_info_raw = pickle.load(open(os.path.join(path_pickles, 'Y_Info.pickle'),'r'))
  print "------------------"
  print "Loaded Annotations"
  print "------------------"

  # Pre-processing on X and y

  ## Fit the Model
  print "-----------------"
  print "Fitting the Model"
  print "-----------------"
  # Select Classifier
  # Fit X and y
  print "-----------------------------"
  print "Model Fitted. Writing out now"
  print "-----------------------------"

  ## Write out the Model
  pickle.dump(model, open(os.path.join(path_pickles, 'Model_1st_Tier.pickle'),'wb'))
