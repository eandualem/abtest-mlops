import mlflow
from config import Config
from df_helper import DfHelper
from sklearn.model_selection import KFold
from evaluate_model import EvaluateModel


Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
helper = DfHelper()

X_train = helper.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = helper.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))


class TrainModel():

  def __init__(self, model, name, params):
    self.model = model
    self.name = name
    self.params = params

  def get_avg_metics(self, metrics_list):
    length = len(metrics_list)

    return dict(
      accuracy=sum([m["accuracy"] for m in metrics_list]) / length,
      r_squared=sum([m["r_squared"] for m in metrics_list]) / length,
      rmse=sum([m["rmse"] for m in metrics_list]) / length,
      mae=sum([m["mae"] for m in metrics_list]) / length,
    )

  def get_optimal_model(self):
    if(self.name == 'XGBoost'):
      mlflow.set_experiment('XGBoost')
      mlflow.xgboost.autolog()
    elif(self.name == 'Decision Tree'):
      mlflow.set_experiment('Decision Tree')
      mlflow.sklearn.autolog()
    else:
      mlflow.set_experiment('Logistic Regression')
      mlflow.sklearn.autolog()
    return self.train_model()

  def train_model(self):
    best_model = None
    best_param = None
    avg_metrics = dict(
      accuracy=0,
      r_squared=0,
      rmse=0,
      mae=0
    )

    for param in self.params:
      new_model = self.model(param)
      metrics = self.train(new_model, 5)
      new_avg_metrics = self.get_avg_metics(metrics)
      if(new_avg_metrics["accuracy"] > avg_metrics["accuracy"]):
        best_model = new_model
        best_param = param
        avg_metrics = new_avg_metrics

    return best_model, best_param, avg_metrics

  def train(self, new_model, num_split):
    kf = KFold(n_splits=num_split)
    randomIter = kf.split(X_train)
    metrics_list = []
    for i in range(5):
      train_index, val_index = next(randomIter)
      _X_train = X_train.iloc[train_index]
      _y_train = y_train.iloc[train_index]

      _X_val = X_train.iloc[val_index]
      _y_val = y_train.iloc[val_index]

      new_model.fit(_X_train, _y_train.to_numpy().ravel())
      y_pred = new_model.predict(_X_val)
      evaluate = EvaluateModel(_y_val, y_pred)
      metrics = evaluate.get_metrics()
      metrics_list.append(metrics)

    return metrics_list
