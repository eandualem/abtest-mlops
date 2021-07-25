import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score


class EvaluateModel():

  def __init__(self, actual, pred):
    self.actual = actual
    self.pred = pred

  def rmse(self):
    return np.sqrt(mean_squared_error(self.actual, self.pred))

  def mae(self):
    return mean_absolute_error(self.actual, self.pred)

  def r2(self):
    return r2_score(self.actual, self.pred)

  def accuracy(self):
    return accuracy_score(self.actual, self.pred)

  def get_metrics(self):
    return dict(
      accuracy=self.accuracy(),
      r_squared=self.r2(),
      rmse=self.rmse(),
      mae=self.mae()
    )
