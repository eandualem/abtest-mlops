import pandas as pd
import numpy as np
import sys
from df_helper import DfHelper
from log import get_logger
import dvc.api
from config import Config
my_logger = get_logger("Create Dataset Versions")

'''
This is a simple script for creating different versions of the data AdSmartABdata.csv
based on argument passed:
  - 0 for all data: doesn't modify the data
  - 1 for browser: creates a new data by removing platform-os column
  - 2 for platform-os: creates a new data by removing browser column
'''

len_args = len(sys.argv)
if(len_args < 2):
  my_logger.exception(
    '''
      You must pass an argument for version of data
      Insert: 
        - 0 for all data
        - 1 for browser
        - 2 for platform-os
    ''')

elif (int(sys.argv[1]) > 2):
  my_logger.exception(
    '''
      There are only 3 options
      Insert: 
        - 0 for all data
        - 1 for browser
        - 2 for platform-os
    ''')

else:
  helper = DfHelper()
  column = int(sys.argv[1])
  np.random.seed(Config.RANDOM_SEED)
  version = 'v4'  # this is the version that contains the whole data
  data_url = dvc.api.get_url(path=str(Config.DATASET_FILE_PATH), repo=str(Config.REPO), rev=version)
  df = pd.read_csv(data_url, sep=',')

  if(column == 0):
    helper.save_csv(df, "../data/AdSmartABdata.csv")

  elif(column == 1):
    df.drop('platform_os', inplace=True, axis=1)
    helper.save_csv(df, "../data/AdSmartABdata.csv")

  elif(column == 2):
    df.drop('browser', inplace=True, axis=1)
    helper.save_csv(df, "../data/AdSmartABdata.csv")
