execfile('Utils.py')
import Utils
import os

if __name__ == "__main__":
  pickler = Utils.pickler()
  dataset_tools = Utils.dataset_tools(Utils.dist(), Utils.nltk_tools(), pickler, Utils.tools())
  y = dataset_tools.prepAnnotations(os.path.join(pickler.pathCode, "Annotations.txt"))
  pickler.dumpPickle(y, "Annotations")
