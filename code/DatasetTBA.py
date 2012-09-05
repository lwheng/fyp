# Precompile files
execfile('Utils.py')
execfile('Citprov.py')

import Citprov
import Utils
import sys

if __name__ == '__main__':
  nltk_Tools = Utils.nltk_tools()
  tools = Utils.tools()
  weight = Utils.weight()
  dist = Utils.dist()
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(tools, dist)

  run = Citprov.citprov(nltk_Tools, tools, weight, dist, pickler)
  raw = pickler.loadPickle(pickler.pathRaw)
  experiment = dataset_tools.fetchExperiment(raw)
  dataset = dataset_tools.prepDataset(run, raw, experiment)
  print len(dataset)
  pickler.dumpPickle(dataset, "DatasetTBA")
