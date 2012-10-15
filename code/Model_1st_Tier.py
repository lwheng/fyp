import cPickle as pickle

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
  # 1. Contexts
  print "----------------"
  print "Loading Contexts"
  print "----------------"
  contexts = pickle.load(open(os.path.join(path_pickles,'Contexts.pickle'),'r'))
  print "---------------------------------"
  print "Loaded Contexts. Extract Features"
  print "---------------------------------"
  feature_extractor = Utils.extract_features()

  # 2. Annotations
  print "-------------------"
  print "Loading Annotations"
  print "-------------------"
