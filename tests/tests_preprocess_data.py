import unittest
from services import PreprocessDataService
import pandas as pd
import numpy as np


class TestPreprocessDataService(unittest.TestCase):
    def setUp(self):
        self.preprocess_data_service = PreprocessDataService()

    def test_preprocess_data(self):
        rawData = pd.DataFrame({
            'sensor_value': [100, 110, None, 115, 120]
            })
        data = self.preprocess_data_service.preprocess_data(rawData)
        self.assertIsNotNone(data)
        self.assertFalse(data.isNull().values.any())

if __name__ == '__main__':
    unittest.main()