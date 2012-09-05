execfile('Utils.py')
import Utils

p = Utils.pickler()
d = Utils.dataset_tools(Utils.dist(), Utils.tools())

authors = p.loadPickle(p.pathAuthors)
experiment = p.loadPickle(p.pathExperiment)
titles = p.loadPickle(p.pathTitles)

raw = d.prepRaw(d.dist, d.tools, authors, experiment, titles)
print len(raw.keys())
p.dumpPickle(raw, "Raw")
