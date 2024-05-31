import unittest 
import pandas as pd
from services import LoadDataService
from sqlmodel import SQLModel

# Test class for LoadDataService
class TestLoadDataService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        self.load_data_service = LoadDataService()

    # Test method to assure data is a pandas dataframe instance and not empty
    def test_load_data(self):
        data = self.load_data_service.read_data('dataset/test.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    '''
    Test method to queck if generated model is an instance of SQLModel and to
    assert if columns are correctly filled
    # TO BE DONE
    '''
    def test_generate_sqlmodel(self):
        sqlmodel = self.load_data_service.generate_sqlmodel('dataset/test.csv')
        self.assertIsInstance(sqlmodel, SQLModel)
        self.assertFalse([column for column in sqlmodel.__table__.columns], [])

if __name__ == '__main__':
    unittest.main()
