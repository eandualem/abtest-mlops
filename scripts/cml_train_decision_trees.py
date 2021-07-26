import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join('./scripts')))
from train_model import TrainModel

'''
  this script is for testing continues model training using github actions
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

with open("metrics.txt", "w") as outfile:
  json.dump(avg_metrics, outfile)
