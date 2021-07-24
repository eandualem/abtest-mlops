import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dvc.api
import mlflow

from config import Config

np.random.seed(Config.RANDOM_SEED)

data_url = dvc.api.get_url(path=str(Config.DATASET_FILE_PATH), repo=str(Config.REPO), rev="v2")

df = pd.read_csv(data_url, sep=',')

# Log data params
mlflow.log_params('data_url', data_url)
mlflow.log_params('input_rows', df.shape[0])
mlflow.log_params('input_cols', df.shape[1])

df_train, df_test = train_test_split(df, test_size=0.1, random_state=Config.RANDOM_SEED,)

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)
