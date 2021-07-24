import pickle
from xgboost import XGBClassifier
from train_model import train_model
from config import Config


def model(solver):
  model = XGBClassifier(solver=solver, random_state=42)
  return model


final_model = train_model(model, "XGBoost")
pickle.dump(final_model, open(str(Config.MODELS_PATH / "xgboost_model.pickle"), "wb"))