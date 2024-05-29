import unittest 
import pandas as pd
from services.load_data import LoadDataService

class TestLoadDataService(unittest.TestCase):
    def setUp(self):
        self.load_data_service = LoadDataService()

    def test_load_data(self):
        data = self.load_data_service.read_data('data/empty.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

if __name__ == '__main__':
    unittest.main()
