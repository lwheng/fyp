from sklearn import svm

class classifier:
  def __init__(self):
    self.data = []
    self.target = []
    # Specify what classifier to use here
    self.clf = svm.SVC()

  def loadData(self, source):
    self.data = source

  def loadTarget(self, source):
    self.target = source

  def prepClassifier(self, data, target):
    # Data: Observations
    # Target: Known classifications
    self.data = data
    self.target = target
    self.clf.fit(data, target)

  def predict(self, observation):
    # Takes in an observation and returns a prediction
    return self.clf.predict(observation)
