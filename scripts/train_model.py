from config import Config
from df_helper import DfHelper
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
helper = DfHelper()

X_train = helper.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = helper.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))


def train_model(_model):
  kf = KFold(n_splits=5)
  solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
  avg_score = 0
  best_model = None

  for solver in solvers:
    scores = []
    model = _model(solver)
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

  # return the best model
  return best_model
