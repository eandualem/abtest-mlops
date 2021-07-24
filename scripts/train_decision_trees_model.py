import pickle
from train_model import train_model
from sklearn.tree import DecisionTreeClassifier
from config import Config
from df_helper import DfHelper
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
helper = DfHelper()

mlflow.set_experiment('Decision Tree')
def eval_metrics(actual, pred):
  rmse = math.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2

X_train = helper.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = helper.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

kf = KFold(n_splits=5)
avg_score = 0
params=[
  {'max_depth':2,'min_samples_split':2},
  {'max_depth':3,'min_samples_split':2},
  {'max_depth':4,'min_samples_split':2},
  {'max_depth':5,'min_samples_split':2},
  {'max_depth':6,'min_samples_split':2}]
best_model = None
best_params = params[0]

mlflow.log_param('Model', "Decision Tree")
mlflow.log_param('Params', params)

for param in params:
    scores = []
    model = DecisionTreeClassifier(random_state=42, **param)
    randomIter = kf.split(X_train)
    for i in range(5):
      train_index, val_index = next(randomIter)
      _X_train = X_train.iloc[train_index]
      _y_train = y_train.iloc[train_index]

      _X_val = X_train.iloc[val_index]
      _y_val = y_train.iloc[val_index]

      model.fit(_X_train, _y_train.to_numpy().ravel())
      y_preds = model.predict(_X_val)
      score = accuracy_score(_y_val, y_preds)
      scores.append(score)

    avg_score_for_solver = sum(scores) / len(scores)
    if(avg_score_for_solver > avg_score):
      avg_score = avg_score_for_solver
      best_model = model
      best_params = param

mlflow.log_param('Best Solver', best_params)
mlflow.log_metric("Average Score", avg_score)
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "Decision Tree", signature=signature)

pickle.dump(best_model, open(str(Config.MODELS_PATH / "decision_tree_model.pickle"), "wb"))