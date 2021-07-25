import pickle
from config import Config
from train_model import TrainModel
from sklearn.tree import DecisionTreeClassifier

'''
  simple script for training decision tree using TrainModel class
'''


def model(param):
  model = DecisionTreeClassifier(random_state=42, **param)
  return model


params = [
    {'max_depth': 2, 'min_samples_split': 2},
    {'max_depth': 3, 'min_samples_split': 2},
    {'max_depth': 4, 'min_samples_split': 2},
    {'max_depth': 5, 'min_samples_split': 2},
    {'max_depth': 6, 'min_samples_split': 2}]

train_model = TrainModel(model, "Decision Tree", params=params)
final_model, best_param, avg_metrics = train_model.get_optimal_model()

pickle.dump(final_model, open(str(Config.MODELS_PATH / "decision_tree_model.pickle"), "wb"))
