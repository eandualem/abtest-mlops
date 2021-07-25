import json
import pickle
import pandas as pd
from config import Config
from evaluate_model import EvaluateModel

'''
This is a simple script for computing evaluation metrics for decision tree model
'''

Config.METRICS_FILE_PATH.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

model = pickle.load(open(str(Config.MODELS_PATH / "decision_tree_model.pickle"), "rb"))

y_pred = model.predict(X_test)
evaluate = EvaluateModel(y_test, y_pred)
metrics = evaluate.get_metrics()

with open(str(Config.METRICS_FILE_PATH / "decision_tree_metrics.json"), "w") as outfile:
  json.dump(metrics, outfile)
