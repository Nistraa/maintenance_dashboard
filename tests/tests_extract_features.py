import unittest
import numpy as np
import pandas as pd
from services import ExtractFeaturesService

'''
Test class for ExtractFeaturesService
'''
class TestExtractFeaturesService(unittest.TestCase):
    '''
    Unittest special method to setup the test with a fresh state and test case
    '''
    def setUp(self):
        self.extract_features_service = ExtractFeaturesService()
        self.data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'],
            'target': [1, 0, 1, 0, 1, 0]        
            })

    '''
    Test method using a dataframe to assure extracted features
    '''
    def test_pandas_standard_method_head(self):
        dataframe = self.extract_features_service.PandasStandardMethods.pandas_standard_method(self, self.data, 'head')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe, self.data.head())

    def test_pandas_standard_method_describe(self):
        dataframe = self.extract_features_service.PandasStandardMethods.pandas_standard_method(self, self.data, 'describe')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe, self.data.describe())

    def test_pandas_standard_method_info(self):
        dataframe = self.extract_features_service.PandasStandardMethods.pandas_standard_method(self, self.data, 'info')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe, self.data.info())

    def test_pandas_standard_method_isna(self):
        dataframe = self.extract_features_service.PandasStandardMethods.pandas_standard_method(self, self.data, 'isna')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe, self.data.isna())
      
    def test_pandas_standard_method_uniques(self):
        dataframe = self.extract_features_service.PandasStandardMethods.pandas_standard_method(self, self.data, 'uniques')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe, self.data.nunique())
      

if __name__ == '__main__':
    unittest.main()