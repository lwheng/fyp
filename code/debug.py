import Utils
p = Utils.pickler()
data = p.dataset
for k in data.keys():
  contexts = data[k]['contexts']
  for c in contexts:
    print c.attributes['citStr'].value
