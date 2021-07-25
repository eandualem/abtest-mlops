import pickle
from config import Config
from train_model import TrainModel
from sklearn.linear_model import LogisticRegression

'''
  simple script for training logistic regression using TrainModel class
'''


def model(param):
  model = LogisticRegression(solver=param, random_state=42)
  return model


params = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

train_model = TrainModel(model, "Logistic Regression", params=params)

final_model, best_param, avg_metrics = train_model.get_optimal_model()

pickle.dump(final_model, open(str(Config.MODELS_PATH / "logistic_model.pickle"), "wb"))
