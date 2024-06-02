import unittest
import pandas as pd
from services import VisualizeDataService

'''
Test class for VisualizeDataService
'''
class TestVisualizeDataService(unittest.TestCase):
    # Unittest special method to setup the test with a fresh state
    def setUp(self):
        self.visualize_data_service = VisualizeDataService()


    '''
    Test method assuring data visualization is handled correctly
    '''
    def test_visualize_data(self):
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='1/1/2020', periods=5), 
            'sensor_value': [100, 110, 105, 115, 120]
        })
        
        result = self.visualize_data_service.visualize_data(data)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()

