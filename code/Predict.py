import Utils
import Feature_Extractor

class predictor:
  def __init__(self):
    self.pickler = Utils.pickler()
    self.model = self.pickler.loadPickle(self.pickler.pathModel)
  
  def predict(self, model, query):
    # Process query
    # Extract features from query, Feature_Extractor.extractFeatures(cite_key, context, citing_col)
    return model.predict(queryProcessed)
