import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from train_model import TrainModel
from config import Config

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

'''
Adding images to report
'''

def plot_confusion_metrics(actual, y_preds):
  plt.figure(figsize=(8, 6))
  cf_matrix = metrics.confusion_matrix(actual, y_preds)
  sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
  plt.title('Confusion matrix', fontsize=15, fontweight='bold')
  plt.ylabel('Actual', fontsize=14)
  plt.xlabel('Predicted', fontsize=14)
  plt.savefig("confusion_matrix.png", dpi=120)
  plt.close() 
  

def plot_feature_importance(feat_imp):
  plt.figure(figsize=(10, 6))
  sns.barplot(x="Feature Importance", y=feat_imp.index, data=feat_imp)
  plt.ylabel('Feature', fontsize=14)
  plt.xlabel('Feature Importance', fontsize=14)
  plt.savefig("feature_importance.png", dpi=120)
  plt.close() 


X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))


y_preds = final_model.predict(X_test)
plot_confusion_metrics(y_test, y_preds)


feature_columns = [f.replace('_', ' ').capitalize() for f in X_test]
feature_columns
feat_imp = pd.DataFrame({'Feature Importance': final_model.feature_importances_})
feat_imp['Feature'] = feature_columns
feat_imp = feat_imp.set_index('Feature')
feat_imp = feat_imp.sort_values(by=['Feature Importance'], ascending=False)

plot_feature_importance(feat_imp)


  

