execfile('Utils.py')
import Utils

p = Utils.pickler()
d = Utils.dataset(Utils.dist(), Utils.tools())

dataset = d.prepDataset(d.dist, d.tools, p.authors, p.experiment, p.titles)
print dataset
print len(dataset.keys())
p.dumpPickle(dataset, "Dataset")
