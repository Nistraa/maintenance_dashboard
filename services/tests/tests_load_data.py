import unittest 
import pandas as pd
from services import LoadDataService

'''
Test class for LoadDataService
'''
class TestLoadDataService(unittest.TestCase):
    '''
    Unittest special method to setup the test with a fresh state
    '''
    def setUp(self):
        self.load_data_service = LoadDataService()

    '''
    Test method to assure data is a pandas dataframe instance and not empty
    '''
    def test_load_data(self):
        data = self.load_data_service.read_data('dataset/test.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

if __name__ == '__main__':
    unittest.main()
