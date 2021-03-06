import os
import sys
import unittest 
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('../scripts')))
from df_helper import DfHelper


class TestDfHelper(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.helper = DfHelper()

    def test_save_csv(self):
        df = pd.DataFrame({'col1': [1, 2, 1], 'col2': [3, 4, 3]})
        self.helper.save_csv(df, './test.csv', False)
        df2 = pd.read_csv('test.csv')
        self.assertEqual(df.shape, df2.shape)
      
    def test_read_csv(self):
        df = self.helper.read_csv('test.csv')
        df2 = pd.read_csv('test.csv')
        self.assertEqual(df.shape, df2.shape)

if __name__ == '__main__':
    unittest.main()
