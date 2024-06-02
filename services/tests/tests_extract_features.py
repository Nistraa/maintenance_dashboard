import unittest
import pandas as pd
import pandas.testing as pdt
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
    Test method using a test dataframe to compare if features were extracted
    correctly. 
    '''
    def test_pandas_standard_method_head(self):
        dataframe = self.extract_features_service.pandas_standard_methods.pandas_standard_method(self.data, 'head')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        pdt.assert_frame_equal(dataframe, self.data.head())

    def test_pandas_standard_method_describe(self):
        dataframe = self.extract_features_service.pandas_standard_methods.pandas_standard_method(self.data, 'describe')

        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        pdt.assert_frame_equal(dataframe, self.data.describe())


    def test_pandas_standard_method_isna(self):
        dataframe = self.extract_features_service.pandas_standard_methods.pandas_standard_method(self.data, 'isna')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
        pdt.assert_frame_equal(dataframe, self.data.isna())
      
    def test_pandas_standard_method_uniques(self):
        dataframe = self.extract_features_service.pandas_standard_methods.pandas_standard_method(self.data, 'uniques')

        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.Series)
        pdt.assert_series_equal(dataframe, self.data.nunique())
    '''
    Test method to assure if dataframe gets succesfully transformed into
    type dataframe after calling .info() method
    '''
    def test_pandas_standard_method_info(self):
        dataframe = self.extract_features_service.pandas_standard_methods.pandas_standard_method(self.data, 'info')
        
        self.assertIsNotNone(dataframe)
        self.assertIsInstance(dataframe, pd.DataFrame)
      

if __name__ == '__main__':
    unittest.main()