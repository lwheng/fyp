# Precompile files
execfile('Utils.py')
execfile('Feature_Extractor.py')

import Feature_Extractor
import Utils
import sys

if __name__ == '__main__':
  nltk_Tools = Utils.nltk_tools()
  tools = Utils.tools()
  weight = Utils.weight()
  dist = Utils.dist()
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(dist, nltk_Tools, pickler, tools)

  run = Feature_Extractor.extractor(dist, nltk_Tools, pickler, tools, weight)
  raw = pickler.loadPickle(pickler.pathRaw)
  experiment = dataset_tools.fetchExperiment(raw)
  #(forannotation, keys, X) = dataset_tools.prepDataset(run, raw, experiment)
  #pickler.dumpPickle(forannotation, "For_Annotation")
  #pickler.dumpPickle(keys, "DatasetTBA_keys")
  #pickler.dumpPickle(X, "DatasetTBA")
  (forannotation, keys, X) = dataset_tools.prepDatasetCFS(run, raw, experiment)
  pickler.dumpPickle(forannotation, "For_AnnotationCFS")
  pickler.dumpPickle(keys, "DatasetTBA_keysCFS")
  pickler.dumpPickle(X, "DatasetTBACFS")
  print X
  print X.shape
