import Utils

class predictor:
  def __init__(self):
    self.pickler = Utils.pickler()
    self.model = self.pickler.loadPickle(self.pickler.pathModel)
  
  def predict(self, model, query):
    # Process query
    queryProcessed = ""
    return model.predict(queryProcessed)
