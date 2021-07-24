import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.MODELS_PATH / "decision_tree_model.pickle"), "wb"))
