import unittest
from services import PreprocessDataService
import pandas as pd
import numpy as np

# Test class for PreprocessDataService
class TestPreprocessDataService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        self.preprocess_data_service = PreprocessDataService()

    # Test method assuring datamissing values are handled
    def test_preprocess_data(self):
        rawData = pd.DataFrame({
            'sensor_value': [100, 110, None, 115, 120]
            })
        data = self.preprocess_data_service.preprocess_data(rawData)
        self.assertIsNotNone(data)
        self.assertFalse(data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()