import unittest
import numpy as np
import pandas as pd
from services.visualize_data import VisualizeDataService

class TestVisualizeDataService(unittest.TestCase):
    def setUp(self):
        self.visualize_data_service = VisualizeDataService()

    def test_visualize_data(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='1/1/2020', periods=5), 
            'sensor_value': [100, 110, 105, 115, 120]})
        result = self.visualize_data_service.visualize_data(data)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()

