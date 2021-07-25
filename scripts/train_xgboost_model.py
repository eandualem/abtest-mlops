import pickle
from config import Config
from xgboost import XGBClassifier
from train_model import TrainModel

'''
  simple script for training XGBoost using TrainModel class
'''

def model(param):
  model = XGBClassifier(random_state=42, eval_metric='logloss', **param)
  return model


params = [
  {'learning_rate': 0.01},
  {'learning_rate': 0.02},
  {'learning_rate': 0.03},
  {'learning_rate': 0.04},
  {'learning_rate': 0.05}]


train_model = TrainModel(model, "XGBoost", params=params)

final_model, best_param, avg_metrics = train_model.get_optimal_model()

pickle.dump(final_model, open(str(Config.MODELS_PATH / "xgboost_model.pickle"), "wb"))
